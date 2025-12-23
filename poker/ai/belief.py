from __future__ import annotations

import random
from typing import Dict, List, Tuple

from poker.helpers.abstraction import (
    GameState,
    Player,
    Act,
    Action,
    BIT,
    compute_feature_mask,
    id_to_card,
    card_to_id,
)
from poker.helpers.cards import make_deck

Hand = Tuple[int, int]
Range = Dict[Hand, float]


def _has(mask: int, name: str) -> bool:
    b = BIT.get(name)
    return False if b is None else ((mask >> b) & 1) == 1


def _remaining_deck_ids(exclude_ids: List[int]) -> List[int]:
    exclude_cards = [id_to_card(x) for x in exclude_ids]
    deck_cards = make_deck(exclude=exclude_cards)
    return [card_to_id(str(c)) for c in deck_cards]


# ---------------------------
# Uniform samplers (baseline)
# ---------------------------

def sample_opponent_hand_uniform(
    gs: GameState,
    my_hand_ids: Tuple[int, int],
    rng: random.Random,
) -> Hand:
    known: List[int] = [my_hand_ids[0], my_hand_ids[1]]
    known.extend([cid for cid in gs.board if cid != -1])
    deck_ids = _remaining_deck_ids(known)
    rng.shuffle(deck_ids)
    return (deck_ids.pop(), deck_ids.pop())


def sample_runout_uniform(
    gs: GameState,
    my_hand_ids: Tuple[int, int],
    opp_hand_ids: Tuple[int, int],
    rng: random.Random,
) -> Tuple[int, int, int, int, int]:
    known: List[int] = [my_hand_ids[0], my_hand_ids[1], opp_hand_ids[0], opp_hand_ids[1]]
    known.extend([cid for cid in gs.board if cid != -1])
    deck_ids = _remaining_deck_ids(known)

    rng.shuffle(deck_ids)

    board_ids = [cid for cid in gs.board if cid != -1]
    while len(board_ids) < 5:
        board_ids.append(deck_ids.pop())
    return (board_ids[0], board_ids[1], board_ids[2], board_ids[3], board_ids[4])


# ---------------------------
# Simple strength scoring
# ---------------------------

def bucket_score(gs: GameState, hand: Hand) -> int:
    """
    Cheap hand strength proxy using your feature mask bits.
    Bigger = stronger.
    """
    mask = compute_feature_mask(hand, gs.board)

    score = 0

    # monsters
    if _has(mask, "made_trips_or_better"):
        score += 500
    if _has(mask, "made_two_pair_or_better"):
        score += 350

    # pair quality
    if _has(mask, "overpair"):
        score += 250
    if _has(mask, "top_pair"):
        score += 200
    if _has(mask, "middle_pair"):
        score += 120
    if _has(mask, "bottom_pair"):
        score += 80
    if _has(mask, "underpair"):
        score += 60

    # draws
    if _has(mask, "combo_draw"):
        score += 160
    if _has(mask, "has_flush_draw"):
        score += 90
    if _has(mask, "has_straight_draw"):
        score += 70

    # overcards as “some equity”
    if _has(mask, "two_overcards"):
        score += 40
    if _has(mask, "one_overcard"):
        score += 20

    return score


def hand_tier_from_score(s: int) -> int:
    """
    Convert bucket_score into a small tier number for easy debugging.
    0 = air/weak, 4 = very strong.
    """
    if s >= 500:
        return 4  # trips+
    if s >= 350:
        return 3  # two pair+
    if s >= 180:
        return 2  # top pair-ish / strong draws
    if s >= 70:
        return 1  # weak pair / some draw equity
    return 0      # air


# ---------------------------
# Range building + weighting
# ---------------------------

def build_uniform_range(gs: GameState, my_hand_ids: Tuple[int, int]) -> Range:
    known: List[int] = [my_hand_ids[0], my_hand_ids[1]]
    known.extend([cid for cid in gs.board if cid != -1])
    deck_ids = _remaining_deck_ids(known)

    r: Range = {}
    for i in range(len(deck_ids)):
        for j in range(i + 1, len(deck_ids)):
            h = (deck_ids[i], deck_ids[j])
            r[h] = 1.0
    return r


def _renormalize(r: Range) -> Range:
    s = float(sum(r.values()))
    if s <= 0:
        return r
    inv = 1.0 / s
    return {h: w * inv for h, w in r.items()}


def sample_opponent_hand_weighted(r: Range, rng: random.Random) -> Hand:
    """
    Weighted random choice over villain hands.
    """
    if not r:
        raise ValueError("Empty range")
    total = float(sum(r.values()))
    if total <= 0:
        raise ValueError("Range weights sum to 0")

    x = rng.random() * total
    acc = 0.0
    for h, w in r.items():
        acc += float(w)
        if acc >= x:
            return h
    return next(iter(r.keys()))


# ---------------------------
# “Range update” from actions
# ---------------------------

# Simple constants: how “tight” villain’s betting is assumed to be
ACTION_TOP_FRAC = {
    Act.BET: 0.80,
    Act.RAISE: 0.30,
    Act.ALLIN: 0.10,
    # CALL/CHECK/FOLD don’t tighten here (we keep them broad)
}

# How aggressively to downweight hands outside the “top fraction”
OUTSIDE_MULT = {
    Act.BET: 1,
    Act.RAISE: 0.8,
    Act.ALLIN: 0.2,
}


def _apply_action_update(gs: GameState, r: Range, act: int) -> Range:
    """
    Core simple idea:
      - score every combo vs this board
      - if villain makes an aggressive action, we bias toward the top X% by score
      - we DO NOT hard-eliminate outside hands; we just heavily downweight them
    """
    top_frac = ACTION_TOP_FRAC.get(act, None)
    if top_frac is None:
        return r  # for CALL/CHECK/etc.

    items = [(h, w, bucket_score(gs, h)) for h, w in r.items()]
    items.sort(key=lambda t: t[2], reverse=True)  # sort by score, best first

    n = len(items)
    if n == 0:
        return r

    cutoff_idx = max(1, int(round(top_frac * n)))
    top_set = set(h for (h, _, _) in items[:cutoff_idx])

    outside_mult = OUTSIDE_MULT.get(act, 0.25)

    new: Range = {}
    for (h, w, _) in items:
        if h in top_set:
            new[h] = w * 1.0
        else:
            new[h] = w * outside_mult

    return _renormalize(new)


def range_from_history_weighted(gs: GameState, my_hand_ids: Tuple[int, int]) -> Range:
    """
    Start uniform, then condition on villain’s own actions in action_history.
    This is the “range updating” mechanism.
    """
    r = _renormalize(build_uniform_range(gs, my_hand_ids))

    for enc in gs.action_history:
        a = Action.decode(enc)
        if int(a.player) != int(Player.VILLAIN):
            continue
        r = _apply_action_update(gs, r, int(a.act))

    return _renormalize(r)


# Keep this name too, since some of your files import it.
def range_from_history_conservative(gs: GameState, my_hand_ids: Tuple[int, int]) -> Range:
    return range_from_history_weighted(gs, my_hand_ids)
