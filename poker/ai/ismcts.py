from __future__ import annotations

from dataclasses import dataclass, field, replace
import random
from typing import Dict, Optional, Tuple, List

from poker.ai.belief import (
    sample_runout_uniform,
    range_from_history_conservative,
    sample_opponent_hand_weighted,
)

from poker.ai.infoset import (
    InfoNode,
    infoset_key,
    ucb_select_action,
    best_action_by_ev,
    bet_bucket_from_amount,
)
from poker.helpers.bucket_policy import suggest_rollout_action
from poker.helpers.cache import LRUTranspoTable
from poker.helpers.abstraction import (
    GameState,
    Player,
    Act,
    Street,
    pad_board,
    BIT,
    compute_feature_mask,
)
from poker.engine.simple_hu_postflop import (
    PublicState,
    legal_actions,
    apply_action,
    is_terminal,
    showdown_result,
)


def _has(mask: int, name: str) -> bool:
    b = BIT.get(name)
    return False if b is None else ((mask >> b) & 1) == 1


def _replace_board_ids(ps: PublicState, board_ids_prefix: List[int]) -> PublicState:
    new_board = pad_board(board_ids_prefix)
    gs2 = replace(ps.gs, board=new_board)

    kwargs = {}
    if hasattr(ps, "raises_this_street"):
        kwargs["raises_this_street"] = getattr(ps, "raises_this_street")

    return PublicState(
        gs=gs2,
        street_bet_open=ps.street_bet_open,
        bet_by=ps.bet_by,
        bet_amount=ps.bet_amount,
        **kwargs,
    )


def _reveal_board_for_street(ps: PublicState, full_board: Tuple[int, int, int, int, int]) -> PublicState:
    street = ps.gs.street
    if street <= 0:
        return ps
    target_len = 3 if street == Street.FLOP else (4 if street == Street.TURN else 5)
    current = [cid for cid in ps.gs.board if cid != -1]
    if len(current) >= target_len:
        return ps
    desired = [cid for cid in full_board if cid != -1][:target_len]
    return _replace_board_ids(ps, desired)


# -------------------------
# Public pressure helpers (no cheating)
# -------------------------

def _facing_bet(ps: PublicState) -> bool:
    return bool(ps.street_bet_open) and int(ps.bet_amount or 0) > 0


def _call_price(ps: PublicState) -> int:
    return int(ps.bet_amount or 0)


def _pot_size(ps: PublicState) -> int:
    return int(ps.gs.pot)


def _spr(ps: PublicState) -> float:
    pot = _pot_size(ps)
    if pot <= 0:
        return 999.0
    eff = min(int(ps.gs.hero_stack), int(ps.gs.villain_stack))
    return eff / float(pot)


def _is_air(mask: int) -> bool:
    pairish_or_better = (
        _has(mask, "made_trips_or_better")
        or _has(mask, "made_two_pair_or_better")
        or _has(mask, "top_pair")
        or _has(mask, "middle_pair")
        or _has(mask, "bottom_pair")
        or _has(mask, "overpair")
        or _has(mask, "underpair")
    )

    drawish = (
        _has(mask, "combo_draw")
        or _has(mask, "has_flush_draw")
        or _has(mask, "has_straight_draw")
        or _has(mask, "is_oesd")
        or _has(mask, "is_gutshot")
    )

    overcards = _has(mask, "two_overcards") or _has(mask, "one_overcard")

    return not (pairish_or_better or drawish or overcards)


@dataclass(slots=True)
class ISMCTS:
    rng: Optional[random.Random] = None
    exploration_c: float = 1.4

    rollout_cache_capacity: int = 200_000
    rollout_cache_min_samples: int = 25

    # more check/call baseline
    rollout_bet_freq: float = 0.30

    # small “call bonus” to avoid insta-folding vs small bets with non-air
    call_bonus_enabled: bool = True
    call_bonus_frac_leq_third_pot: float = 0.08  # *pot chips
    call_bonus_frac_leq_half_pot: float = 0.05   # *pot chips

    # optional fold tax (off by default)
    fold_tax_enabled: bool = False
    fold_tax_vs_leq_half_pot: float = 0.0

    # hard disable ALLIN everywhere (tree + rollouts)
    disable_allin: bool = True

    table: Dict[object, InfoNode] = field(default_factory=dict)
    tt: LRUTranspoTable = field(default_factory=lambda: LRUTranspoTable(capacity=200_000))

    last_root_visits: Dict[int, int] = field(default_factory=dict)
    last_root_q: Dict[int, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.rng = self.rng or random.Random()
        self.tt = LRUTranspoTable(capacity=int(self.rollout_cache_capacity))

    # -------------------------
    # Incentives
    # -------------------------
    def _call_bonus(self, ps: PublicState, mask: int) -> float:
        if not self.call_bonus_enabled:
            return 0.0
        if not _facing_bet(ps) or _is_air(mask):
            return 0.0

        pot = _pot_size(ps)
        call_amt = _call_price(ps)
        if pot <= 0 or call_amt <= 0:
            return 0.0

        frac = call_amt / float(pot)
        if frac <= (1.0 / 3.0):
            return float(self.call_bonus_frac_leq_third_pot) * float(pot)
        if frac <= 0.5:
            return float(self.call_bonus_frac_leq_half_pot) * float(pot)
        return 0.0

    def _fold_tax(self, ps: PublicState) -> float:
        if not self.fold_tax_enabled:
            return 0.0
        if not _facing_bet(ps):
            return 0.0
        pot = _pot_size(ps)
        call_amt = _call_price(ps)
        if pot <= 0 or call_amt <= 0:
            return 0.0
        frac = call_amt / float(pot)
        return float(self.fold_tax_vs_leq_half_pot) if frac <= 0.5 else 0.0

    def _infoset_key_for_ps(self, ps: PublicState, my_hand_ids: Tuple[int, int]):
        fb = 1 if _facing_bet(ps) else 0
        bb = bet_bucket_from_amount(_pot_size(ps), _call_price(ps)) if fb else 0
        return infoset_key(ps.gs, my_hand_ids, facing_bet=fb, bet_bucket=bb)

    def _strip_illegal_or_disabled(self, acts: List[int]) -> List[int]:
        if not self.disable_allin:
            return acts
        return [a for a in acts if a != Act.ALLIN]

    def search(
        self,
        root_state: PublicState,
        my_hand_ids: Tuple[int, int],
        iters: int = 2000,
    ) -> Tuple[int, Dict[int, float]]:
        self.last_root_visits = {}
        self.last_root_q = {}

        root_acts = list(legal_actions(root_state))
        root_acts = self._strip_illegal_or_disabled(root_acts)
        if not root_acts:
            raise ValueError("No legal actions at root (after filtering)")

        vill_range = range_from_history_conservative(root_state.gs, my_hand_ids)

        for _ in range(int(iters)):
            opp = sample_opponent_hand_weighted(vill_range, self.rng)
            full_board = sample_runout_uniform(root_state.gs, my_hand_ids, opp, self.rng)
            self._iterate(root_state, my_hand_ids, opp, full_board)

        I = self._infoset_key_for_ps(root_state, my_hand_ids)
        node = self.table.get(I, InfoNode())
        best, q, n = best_action_by_ev(node, root_acts)

        self.last_root_visits = dict(n)
        self.last_root_q = dict(q)
        return best, q

    def _iterate(
        self,
        ps: PublicState,
        my_hand_ids: Tuple[int, int],
        opp_hand_ids: Tuple[int, int],
        full_board: Tuple[int, int, int, int, int],
    ) -> float:
        ps = _reveal_board_for_street(ps, full_board)

        if is_terminal(ps):
            hero_ev = self._hero_terminal_ev(ps.gs, my_hand_ids, opp_hand_ids, full_board)
            return hero_ev if ps.gs.to_act == Player.HERO else -hero_ev

        acts = list(legal_actions(ps))
        acts = self._strip_illegal_or_disabled(acts)

        if not acts:
            hero_ev = self._hero_terminal_ev(ps.gs, my_hand_ids, opp_hand_ids, full_board)
            return hero_ev if ps.gs.to_act == Player.HERO else -hero_ev

        mask = compute_feature_mask(my_hand_ids, ps.gs.board)

        # never fold monsters (abstraction safety)
        if Act.FOLD in acts and (_has(mask, "made_two_pair_or_better") or _has(mask, "made_trips_or_better")):
            acts = [a for a in acts if a != Act.FOLD]

        I = self._infoset_key_for_ps(ps, my_hand_ids)
        node = self.table.get(I)
        if node is None:
            node = InfoNode()
            self.table[I] = node
        node.ensure_actions(acts)

        fold_tax = self._fold_tax(ps)
        call_bonus = self._call_bonus(ps, mask)

        untried = [a for a in acts if node.na[a] == 0]
        if untried:
            a = untried[self.rng.randrange(len(untried))]
            next_ps = _reveal_board_for_street(apply_action(ps, a), full_board)
            v = self._rollout_value(next_ps, my_hand_ids, opp_hand_ids, full_board)

            if a == Act.CALL and call_bonus != 0.0:
                v += call_bonus
            if a == Act.FOLD and fold_tax != 0.0:
                v -= fold_tax

            node.n += 1
            node.na[a] += 1
            node.wa[a] += v
            return v

        a = ucb_select_action(node, acts, self.exploration_c)
        next_ps = _reveal_board_for_street(apply_action(ps, a), full_board)

        if next_ps.gs.to_act == ps.gs.to_act:
            child_val = self._iterate(next_ps, my_hand_ids, opp_hand_ids, full_board)
        else:
            child_val = -self._iterate(next_ps, opp_hand_ids, my_hand_ids, full_board)

        if a == Act.CALL and call_bonus != 0.0:
            child_val += call_bonus
        if a == Act.FOLD and fold_tax != 0.0:
            child_val -= fold_tax

        node.n += 1
        node.na[a] += 1
        node.wa[a] += child_val
        return child_val

    def _rollout_value(
        self,
        ps: PublicState,
        my_hand_ids: Tuple[int, int],
        opp_hand_ids: Tuple[int, int],
        full_board: Tuple[int, int, int, int, int],
    ) -> float:
        I = self._infoset_key_for_ps(ps, my_hand_ids)
        cached = self.tt.get(I)
        if cached is not None and cached.n >= int(self.rollout_cache_min_samples):
            hero_ev = cached.mean_ev
            return hero_ev if ps.gs.to_act == Player.HERO else -hero_ev

        hero_ev = self._rollout_hero_perspective(ps, my_hand_ids, opp_hand_ids, full_board)
        self.tt.update(I, hero_ev)
        return hero_ev if ps.gs.to_act == Player.HERO else -hero_ev

    # -------------------------
    # Better rollouts, no ALLIN
    # -------------------------
    def _choose_rollout_action(self, ps: PublicState, priv_hand: Tuple[int, int], acts: List[int]) -> int:
        acts = self._strip_illegal_or_disabled(acts)

        mask = compute_feature_mask(priv_hand, ps.gs.board)

        fb = _facing_bet(ps)
        pot = _pot_size(ps)
        call_amt = _call_price(ps)
        frac = (call_amt / float(pot)) if (fb and pot > 0) else 0.0

        strong_made = _has(mask, "made_two_pair_or_better") or _has(mask, "made_trips_or_better")
        has_pairish = (
            _has(mask, "top_pair")
            or _has(mask, "middle_pair")
            or _has(mask, "bottom_pair")
            or _has(mask, "overpair")
            or _has(mask, "underpair")
        )
        has_drawish = (
            _has(mask, "combo_draw")
            or _has(mask, "has_flush_draw")
            or _has(mask, "has_straight_draw")
            or _has(mask, "is_oesd")
            or _has(mask, "is_gutshot")
        )
        air = _is_air(mask)

        if fb:
            if Act.CALL in acts:
                if strong_made or has_pairish or has_drawish:
                    return Act.CALL
                if frac <= 0.33 and self.rng.random() < 0.35:
                    return Act.CALL
                if frac <= 0.50 and self.rng.random() < 0.15:
                    return Act.CALL
            if Act.FOLD in acts:
                return Act.FOLD
            return acts[0]

        # no bet faced: mostly check
        if Act.CHECK in acts and self.rng.random() < 0.70:
            return Act.CHECK

        if Act.BET in acts:
            if strong_made:
                return Act.BET
            if has_drawish and self.rng.random() < 0.60:
                return Act.BET
            if has_pairish and self.rng.random() < 0.40:
                return Act.BET
            if air and self.rng.random() < 0.10:
                return Act.BET

        if Act.RAISE in acts:
            # rollouts: keep raises rare
            if strong_made and self.rng.random() < 0.20:
                return Act.RAISE
            if has_drawish and self.rng.random() < 0.08:
                return Act.RAISE

        # fallback to bucket policy (but never allow ALLIN)
        suggested = suggest_rollout_action(ps.gs, priv_hand, rng=self.rng)
        if suggested.act == Act.ALLIN and self.disable_allin:
            # downgrade
            if Act.BET in acts:
                return Act.BET
            if Act.CHECK in acts:
                return Act.CHECK
            if Act.CALL in acts:
                return Act.CALL
            return acts[0]

        if suggested.act in acts:
            if suggested.act == Act.BET and self.rng.random() > float(self.rollout_bet_freq) and Act.CHECK in acts:
                return Act.CHECK
            return suggested.act

        if Act.CHECK in acts:
            return Act.CHECK
        if Act.CALL in acts:
            return Act.CALL
        return acts[0]

    def _rollout_hero_perspective(
        self,
        state: PublicState,
        hero_hand_ids: Tuple[int, int],
        vill_hand_ids: Tuple[int, int],
        full_board: Tuple[int, int, int, int, int],
    ) -> float:
        ps = _reveal_board_for_street(state, full_board)

        while not is_terminal(ps):
            acts = list(legal_actions(ps))
            acts = self._strip_illegal_or_disabled(acts)
            if not acts:
                break

            priv = hero_hand_ids if ps.gs.to_act == Player.HERO else vill_hand_ids
            act = self._choose_rollout_action(ps, priv, acts)

            ps = _reveal_board_for_street(apply_action(ps, act), full_board)

        return self._hero_terminal_ev(ps.gs, hero_hand_ids, vill_hand_ids, full_board)

    def _hero_terminal_ev(
        self,
        gs: GameState,
        hero_hand_ids: Tuple[int, int],
        vill_hand_ids: Tuple[int, int],
        full_board: Tuple[int, int, int, int, int],
    ) -> float:
        if gs.meta and gs.meta.get("terminal", False):
            w = gs.meta.get("winner", None)
            if w is not None:
                return self._hero_chip_ev_terminal(gs, hero_wins=(int(w) == Player.HERO))

        cmp = showdown_result(hero_hand_ids, vill_hand_ids, full_board)
        if cmp > 0:
            return self._hero_chip_ev_terminal(gs, hero_wins=True)
        if cmp < 0:
            return self._hero_chip_ev_terminal(gs, hero_wins=False)
        return self._hero_chip_ev_terminal(gs, hero_wins=None)

    def _hero_chip_ev_terminal(self, gs: GameState, hero_wins: Optional[bool]) -> float:
        meta = gs.meta or {}
        hero_start = int(meta["hero_start"])

        pot = int(gs.pot)
        hero_stack = int(gs.hero_stack)

        if hero_wins is True:
            hero_final = hero_stack + pot
        elif hero_wins is False:
            hero_final = hero_stack
        else:
            hero_final = hero_stack + (pot // 2)

        return float(hero_final - hero_start)
