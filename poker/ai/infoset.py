from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, Hashable, Iterable, Tuple, Sequence, List

from poker.helpers.abstraction import GameState, BIT, features_to_bitmask, id_to_card
from poker.helpers.features import extract_features
from poker.helpers.texture import board_texture


@dataclass(slots=True)
class InfoNode:
    """
    Bandit-style ISMCTS statistics for an information set I.

      n      = N(I)
      na[a]  = N(I,a)
      wa[a]  = W(I,a) = sum of sampled returns for action a
    """
    n: int = 0
    na: Dict[int, int] = field(default_factory=dict)
    wa: Dict[int, float] = field(default_factory=dict)

    def ensure_actions(self, acts: Iterable[int]) -> None:
        for a in acts:
            if a not in self.na:
                self.na[a] = 0
                self.wa[a] = 0.0


def _bucketize_float(x: float, cutoffs: Sequence[float]) -> int:
    # IMPORTANT: cutoffs is a sequence, not a one-shot iterator
    for i, c in enumerate(cutoffs):
        if x <= c:
            return i
    return len(cutoffs)


def _bucketize_int(x: int, cutoffs: Sequence[int]) -> int:
    for i, c in enumerate(cutoffs):
        if x <= c:
            return i
    return len(cutoffs)


def bet_bucket_from_amount(pot: int, bet_amount: int) -> int:
    """
    Bucket bet size in a simple public way.
    Uses bet/pot ratio:
      0: none / not facing bet
      1: <= 1/4 pot
      2: <= 1/3 pot
      3: <= 1/2 pot
      4: <= 3/4 pot
      5: <= pot
      6: > pot
    """
    if pot <= 0 or bet_amount <= 0:
        return 0
    frac = bet_amount / float(pot)
    if frac <= 0.25:
        return 1
    if frac <= (1.0 / 3.0):
        return 2
    if frac <= 0.5:
        return 3
    if frac <= 0.75:
        return 4
    if frac <= 1.0:
        return 5
    return 6


def public_key(gs: GameState, facing_bet: int = 0, bet_bucket: int = 0) -> Tuple[int, int, int, int, int, int]:
    """
    Public-only key:
      (street, board_texture_mask, spr_bucket, pot_bucket, facing_bet, bet_bucket)

    IMPORTANT: mask is derived ONLY from board texture features, not from hole cards.
    """
    board_cards = [id_to_card(cid) for cid in gs.board if cid != -1]
    tex = board_texture(board_cards) if len(board_cards) >= 3 else {}

    # Only keep board-related flags that exist in BIT
    tex_only = {k: bool(tex.get(k, False)) for k in tex.keys() if k in BIT}
    mask = features_to_bitmask(tex_only) if tex_only else 0

    eff_stack = min(gs.hero_stack, gs.villain_stack)
    spr = (eff_stack / gs.pot) if gs.pot > 0 else 999.0
    spr_bucket = _bucketize_float(spr, [1, 2, 4, 8, 16, 32])

    pot_bucket = _bucketize_int(gs.pot, [10, 25, 50, 100, 200, 400, 800, 1600, 3200])

    return (gs.street, mask, spr_bucket, pot_bucket, int(facing_bet), int(bet_bucket))


def private_strength_bucket(gs: GameState, my_hand_ids: Tuple[int, int]) -> int:
    """
    Coarse private bucket for ISMCTS keys.

    Buckets (0..6):
      0 air / nothing
      1 weak pair
      2 top pair or overpair
      3 two-pair+
      4 flush/straight draw
      5 combo draw
      6 trips+
    """
    board_cards = [id_to_card(cid) for cid in gs.board if cid != -1]
    if len(board_cards) < 3:
        return 0

    h0, h1 = id_to_card(my_hand_ids[0]), id_to_card(my_hand_ids[1])
    feats = extract_features([h0, h1], board_cards)

    made_rank = int(feats.get("made_hand_rank", 0))
    if made_rank >= 3:
        return 6
    if made_rank >= 2:
        return 3

    if bool(feats.get("overpair", False)) or bool(feats.get("top_pair", False)):
        return 2
    if bool(feats.get("middle_pair", False)) or bool(feats.get("bottom_pair", False)) or bool(feats.get("underpair", False)):
        return 1

    has_fd = bool(feats.get("has_flush_draw", False))
    has_sd = bool(feats.get("has_straight_draw", False))
    combo = bool(feats.get("combo_draw", False))
    if combo:
        return 5
    if has_fd or has_sd:
        return 4

    return 0


def infoset_key(
    gs: GameState,
    my_hand_ids: Tuple[int, int],
    *,
    facing_bet: int = 0,
    bet_bucket: int = 0,
) -> Hashable:
    """
    Information-set key for the player-to-act:

      I = (public_key, private_bucket, to_act)

    facing_bet + bet_bucket are PUBLIC, but matter a lot for “same board different pressure”.
    """
    return (
        public_key(gs, facing_bet=facing_bet, bet_bucket=bet_bucket),
        private_strength_bucket(gs, my_hand_ids),
        int(gs.to_act),
    )


def ucb_select_action(node: InfoNode, acts: Iterable[int], exploration_c: float) -> int:
    node.ensure_actions(acts)

    # Try unvisited actions first
    for a in acts:
        if node.na[a] == 0:
            return a

    logN = math.log(node.n + 1.0)
    best_a = None
    best_s = -1e100
    for a in acts:
        mean = node.wa[a] / node.na[a]
        bonus = float(exploration_c) * math.sqrt(logN / node.na[a])
        s = mean + bonus
        if s > best_s:
            best_s = s
            best_a = a
    assert best_a is not None
    return best_a


def best_action_by_ev(node: InfoNode, acts: Iterable[int]) -> Tuple[int, Dict[int, float], Dict[int, int]]:
    node.ensure_actions(acts)
    q: Dict[int, float] = {}
    n: Dict[int, int] = {}
    for a in acts:
        n[a] = node.na[a]
        q[a] = 0.0 if node.na[a] == 0 else node.wa[a] / node.na[a]
    best = max(list(acts), key=lambda a: (q.get(a, 0.0), n.get(a, 0)))
    return best, q, n
