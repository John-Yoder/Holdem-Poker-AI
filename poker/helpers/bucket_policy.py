from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Optional, Tuple

from .abstraction import GameState, BIT, Act, compute_feature_mask, compute_public_bucket_key


@dataclass(slots=True)
class PolicyAction:
    act: int              # Act enum value
    size_bucket: int = 0  # used for BET/RAISE, ignored otherwise


def _has(mask: int, name: str) -> bool:
    b = BIT.get(name)
    return False if b is None else ((mask >> b) & 1) == 1


def suggest_rollout_action(
    state: GameState,
    hero_hand_ids: Tuple[int, int],
    rng: Optional[random.Random] = None
) -> PolicyAction:
    """
    Cheap heuristic rollout policy based on:
      - made-hand-ish signals (pairs/overpair)
      - draws (FD/OESD/gutshot/combo)
      - board danger (flush/straight available)
      - SPR bucket from state.bucket_key()

    This does NOT know legal actions. Your game engine should filter.
    """
    rng = rng or random.Random()

    mask = compute_feature_mask(hero_hand_ids, state.board)
    street, _, spr_bucket, _ = compute_public_bucket_key(state)

    # made hand signals
    made_two_pair_plus = (
        _has(mask, "two_pair")
        or _has(mask, "trips")
        or _has(mask, "full_house")
        or _has(mask, "quads")
    )

    strong_pair = _has(mask, "overpair") or _has(mask, "top_pair")
    weak_pair = _has(mask, "middle_pair") or _has(mask, "bottom_pair")
    draw = _has(mask, "has_flush_draw") or _has(mask, "has_straight_draw")
    combo = _has(mask, "combo_draw")
    two_over = _has(mask, "two_overcards")

    # board danger
    flushy = _has(mask, "flush_available") or _has(mask, "flush_draw_available")
    straighty = _has(mask, "straight_available") or _has(mask, "straight_draw_available")
    dangerous = flushy or straighty

    # sizing bucket guideline:
    # 0=small, 1=mid, 2=big
    def size_for_pressure() -> int:
        if spr_bucket <= 1:
            return 2  # low SPR => bigger pressure / commit
        if spr_bucket <= 3:
            return 1
        return 0

    # --- decision rules ---

    # 0) very strong made hands: push value/protection in rollouts
    if made_two_pair_plus:
        return PolicyAction(act=Act.BET, size_bucket=size_for_pressure())

    # 1) combo draws: play aggressively some of the time
    if combo:
        return PolicyAction(act=Act.BET, size_bucket=size_for_pressure())

    # 2) strong pair: often bet, sometimes check (esp. dangerous boards)
    if strong_pair:
        if dangerous and rng.random() < 0.35:
            return PolicyAction(act=Act.CHECK)
        return PolicyAction(act=Act.BET, size_bucket=size_for_pressure())

    # 3) draw only: mix bet/check based on SPR and danger
    if draw:
        if dangerous and rng.random() < 0.55:
            return PolicyAction(act=Act.CHECK)
        return PolicyAction(act=Act.BET, size_bucket=0 if spr_bucket >= 4 else 1)

    # 4) weak pair: mostly check/call-ish behavior; in rollouts just check
    if weak_pair:
        return PolicyAction(act=Act.CHECK)

    # 5) two overs: stab sometimes on safe boards
    if two_over and not dangerous and rng.random() < 0.35:
        return PolicyAction(act=Act.BET, size_bucket=0)

    # default: check (or fold if your engine says facing a bet)
    return PolicyAction(act=Act.CHECK)
