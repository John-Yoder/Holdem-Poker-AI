from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, Optional, Sequence, Tuple, Union

from .cards import Card, parse_cards, RANK_TO_VAL
from .features import extract_features

from poker.helpers.texture import board_texture

# ------------------------------------------------------------
# Card encoding: 0..51 (rank-major, suit-minor)
# rank 2..A => 0..12, suit s/h/d/c => 0..3
# ------------------------------------------------------------

_SUITS = "shdc"
_SUIT_TO_I = {s: i for i, s in enumerate(_SUITS)}
_I_TO_SUIT = {i: s for s, i in _SUIT_TO_I.items()}

_VAL_TO_RANKI = {v: (v - 2) for v in range(2, 15)}  # 2..14 -> 0..12
_RANKI_TO_VAL = {i: (i + 2) for i in range(13)}     # 0..12 -> 2..14


def card_to_id(c: Union[str, Card]) -> int:
    c = c if isinstance(c, Card) else Card.from_str(c)
    r_i = _VAL_TO_RANKI[c.val]
    s_i = _SUIT_TO_I[c.suit]
    return r_i * 4 + s_i


def id_to_card(cid: int) -> Card:
    if not (0 <= cid < 52):
        raise ValueError(f"Card id out of range: {cid}")
    r_i = cid // 4
    s_i = cid % 4
    return Card(_RANKI_TO_VAL[r_i], _I_TO_SUIT[s_i])


def pad_board(board_ids: Sequence[int]) -> Tuple[int, int, int, int, int]:
    """
    Store board as fixed-length tuple of 5 ints (pad with -1).
    This makes hashing fast and consistent for transposition tables.
    """
    if len(board_ids) > 5:
        raise ValueError("Board cannot exceed 5 cards")
    padded = list(board_ids) + [-1] * (5 - len(board_ids))
    return (padded[0], padded[1], padded[2], padded[3], padded[4])


# ------------------------------------------------------------
# Action encoding (simple and extensible)
# You can later add bet sizing buckets, timestamps, etc.
# ------------------------------------------------------------

class Street(IntEnum):
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3


class Act(IntEnum):
    FOLD = 0
    CHECK = 1
    CALL = 2
    BET = 3
    RAISE = 4
    ALLIN = 5


class Player(IntEnum):
    HERO = 0
    VILLAIN = 1
    # later: multiway => add more players or store player_id int


@dataclass(frozen=True, slots=True)
class Action:
    """
    Compact action record. Keep it small; this repeats a lot.
    amount can be:
      - chips (int)
      - or a sizing bucket id (preferred later)
    """
    street: int      # Street enum value
    player: int      # Player enum value or seat index
    act: int         # Act enum value
    amount: int = 0  # 0 for check/fold etc.

    def encode(self) -> int:
        """
        Packs into a single int for cheap storage:
          bits: [street:2][player:6][act:3][amount:21]  (fits in 32b-ish if amount <= 2M)
        Adjust later if you need bigger.
        """
        if self.amount < 0 or self.amount >= (1 << 21):
            raise ValueError("amount out of encodable range (0..2,097,151)")
        return (self.street & 0b11) << 30 | (self.player & 0b111111) << 24 | (self.act & 0b111) << 21 | (self.amount & ((1 << 21) - 1))

    @staticmethod
    def decode(x: int) -> "Action":
        street = (x >> 30) & 0b11
        player = (x >> 24) & 0b111111
        act = (x >> 21) & 0b111
        amount = x & ((1 << 21) - 1)
        return Action(street=street, player=player, act=act, amount=amount)


# ------------------------------------------------------------
# Feature bitset: stable + fast + extendable
# ------------------------------------------------------------

# Bit positions (add more later; don’t reorder existing ones once you start training/caching)
BIT = {
    # board texture
    "paired_board": 0,
    "trips_on_board": 1,
    "flush_available": 2,
    "flush_draw_available": 3,
    "straight_available": 4,
    "straight_draw_available": 5,
    "ace_high_board": 6,
    "king_high_board": 7,
    "queen_high_board": 8,
    "ten_high_or_lower_board": 9,
    "monotone_flop": 10,
    "two_tone_flop": 11,

    # made-hand relative features
    "top_pair": 16,
    "middle_pair": 17,
    "bottom_pair": 18,
    "overpair": 19,
    "underpair": 20,
    "two_overcards": 21,
    "one_overcard": 22,

    # made-hand tiers (coarser than exact rank/category)
    "made_two_pair_or_better": 23,
    "made_trips_or_better": 24,

    # draw features
    "has_flush_draw": 28,
    "has_straight_draw": 29,
    "is_oesd": 30,
    "is_gutshot": 31,
    "combo_draw": 32,

    # “combo-ish” (optional)
    # (pair_plus_draw exists in earlier monolith; if you re-add it, give it a new bit)
}


def features_to_bitmask(feats: Dict[str, object]) -> int:
    mask = 0
    for k, b in BIT.items():
        v = feats.get(k, False)
        if isinstance(v, bool) and v:
            mask |= (1 << b)
    return mask


# ------------------------------------------------------------
# GameState: compact, hashable, scalable
# ------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class GameState:
    """
    A scalable "known information package" for one decision point.

    Design goals:
      - small + hashable (for transposition tables / MCTS)
      - fixed-size where possible (board padded to 5)
      - extensible via meta (but keep meta small; it breaks determinism if abused)
    """
    # PUBLIC cards (encoded 0..51)
    #
    # IMPORTANT: hole cards are PRIVATE and must NOT live here.
    # This is the core “stop omniscience” refactor that enables ISMCTS.
    board: Tuple[int, int, int, int, int]  # padded with -1 for missing
    street: int                             # Street enum value

    # core numeric context (you'll absolutely need these in poker)
    pot: int
    hero_stack: int
    villain_stack: int
    to_act: int                             # Player enum value or seat index

    # compressed action history
    action_history: Tuple[int, ...] = field(default_factory=tuple)

    # future: opponent model, exploit stats, etc.
    # keep it optional; don’t let it become your primary state.
    meta: Optional[Dict[str, object]] = None

    # ---------- constructors ----------
    @staticmethod
    def from_cards(
        board: Sequence[Union[str, Card]],
        street: Union[int, Street],
        pot: int,
        hero_stack: int,
        villain_stack: int,
        to_act: Union[int, Player],
        action_history: Sequence[Union[int, Action]] = (),
        meta: Optional[Dict[str, object]] = None,
    ) -> "GameState":
        """Build a PUBLIC game state. Hole cards are intentionally excluded."""
        b = parse_cards(board)
        b_ids = [card_to_id(x) for x in b]
        ah = tuple(a if isinstance(a, int) else a.encode() for a in action_history)
        return GameState(
            board=pad_board(b_ids),
            street=int(street),
            pot=int(pot),
            hero_stack=int(hero_stack),
            villain_stack=int(villain_stack),
            to_act=int(to_act),
            action_history=ah,
            meta=meta,
        )

    # ---------- update helpers ----------
    def apply_action(
        self,
        action: Action,
        pot_delta: int = 0,
        hero_stack_delta: int = 0,
        villain_stack_delta: int = 0,
    ) -> "GameState":
        """Functional update: append action + apply deltas."""
        new_hist = self.action_history + (action.encode(),)
        return GameState(
            board=self.board,
            street=self.street,
            pot=self.pot + pot_delta,
            hero_stack=self.hero_stack + hero_stack_delta,
            villain_stack=self.villain_stack + villain_stack_delta,
            to_act=self.to_act,  # engine swaps this
            action_history=new_hist,
            meta=self.meta,
        )

    def with_board(self, new_board: Sequence[Union[str, Card]]) -> "GameState":
        b = parse_cards(new_board)
        b_ids = [card_to_id(x) for x in b]
        return GameState(
            board=pad_board(b_ids),
            street=self.street,
            pot=self.pot,
            hero_stack=self.hero_stack,
            villain_stack=self.villain_stack,
            to_act=self.to_act,
            action_history=self.action_history,
            meta=self.meta,
        )


@dataclass(frozen=True, slots=True)
class PrivateInfo:
    """Private hole cards for a single player."""
    hand: Tuple[int, int]

    @staticmethod
    def from_cards(hand: Sequence[Union[str, Card]]) -> "PrivateInfo":
        h = parse_cards(hand)
        if len(h) != 2:
            raise ValueError("hand must have exactly 2 cards")
        return PrivateInfo(hand=(card_to_id(h[0]), card_to_id(h[1])))


# NOTE: feature extraction and bucketing now live in free functions below,
# because they depend on PRIVATE hole cards.


def _bucketize_float(x: float, cutoffs: Sequence[float]) -> int:
    for i, c in enumerate(cutoffs):
        if x <= c:
            return i
    return len(cutoffs)


def _bucketize_int(x: int, cutoffs: Sequence[int]) -> int:
    for i, c in enumerate(cutoffs):
        if x <= c:
            return i
    return len(cutoffs)


# ------------------------------------------------------------
# Public/private feature & bucketing helpers
# ------------------------------------------------------------

def compute_feature_dict(hand_ids: Tuple[int, int], board_ids: Tuple[int, int, int, int, int]) -> Dict[str, object]:
    """Full (private) feature dict derived from (hand, board)."""
    h0, h1 = id_to_card(hand_ids[0]), id_to_card(hand_ids[1])
    board_cards = [id_to_card(cid) for cid in board_ids if cid != -1]
    if len(board_cards) < 3:
        return {}
    return extract_features([h0, h1], board_cards)


def compute_feature_mask(hand_ids: Tuple[int, int], board_ids: Tuple[int, int, int, int, int]) -> int:
    feats = compute_feature_dict(hand_ids, board_ids)
    return features_to_bitmask(feats) if feats else 0


def compute_public_mask(board_ids):
    board_cards = [id_to_card(cid) for cid in board_ids if cid != -1]
    if len(board_cards) < 3:
        return 0

    tex = board_texture(board_cards) or {}
    tex_only = {k: bool(tex.get(k, False)) for k in tex.keys() if k in BIT}
    return features_to_bitmask(tex_only) if tex_only else 0


def compute_public_bucket_key(gs: GameState) -> Tuple[int, int, int, int]:
    """Bucket key that depends ONLY on public info."""
    mask = compute_public_mask(gs.board)
    eff_stack = min(gs.hero_stack, gs.villain_stack)
    spr = (eff_stack / gs.pot) if gs.pot > 0 else 999.0
    spr_bucket = _bucketize_float(spr, [1, 2, 4, 8, 16, 32])
    pot_bucket = _bucketize_int(gs.pot, [10, 25, 50, 100, 200, 400, 800, 1600, 3200])
    return (gs.street, mask, spr_bucket, pot_bucket)


def private_strength_bucket(mask: int) -> int:
    """Coarse private bucket for infosets (small integer)."""
    def has(name: str) -> bool:
        b = BIT.get(name)
        return False if b is None else ((mask >> b) & 1) == 1

    # made hands
    if has("made_trips_or_better"):
        return 6
    if has("made_two_pair_or_better"):
        return 5
    if has("top_pair") or has("overpair"):
        return 4
    if has("middle_pair") or has("bottom_pair") or has("underpair"):
        return 3
    # draws
    if has("combo_draw"):
        return 2
    if has("has_flush_draw") or has("has_straight_draw"):
        return 1
    return 0
