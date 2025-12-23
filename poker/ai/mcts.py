# Old code no longer used

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Dict, Optional, Tuple

from ..engine.simple_hu_postflop import (
    PublicState,
    legal_actions,
    apply_action,
    is_terminal,
    terminal_winner,
    showdown_result,
)
from ..helpers.abstraction import Player, Act, GameState, pad_board
from ..helpers.bucket_policy import suggest_rollout_action
from ..helpers.cache import LRUTranspoTable


@dataclass(slots=True)
class Node:
    state: PublicState
    parent: Optional["Node"] = None
    parent_action: Optional[int] = None  # Act enum value that led to this node

    children: Dict[int, "Node"] = field(default_factory=dict)  # act -> child node
    n: int = 0                 # visit count
    w: float = 0.0             # total value from CURRENT PLAYER perspective (negamax)

    def q(self) -> float:
        return 0.0 if self.n == 0 else self.w / self.n


class MCTS:
    """
    Minimal MCTS for the simplified HU postflop environment.

    VALUE CONVENTIONS (negamax):
      - Node.w / Node.q are stored from the perspective of the player to act at that node (state.gs.to_act).
      - Rollouts compute a terminal payoff in HERO chip units, then convert to the node player's perspective.
      - During backup, value is negated iff the player-to-act switches between nodes.
    """

    def __init__(
        self,
        rng: Optional[random.Random] = None,
        exploration_c: float = 1.4,
        rollout_cache_capacity: int = 200_000,
        rollout_cache_min_samples: int = 25,
        rollout_bet_freq: float = 0.55,
    ):
        self.rng = rng or random.Random()
        self.c = float(exploration_c)
        self.tt = LRUTranspoTable(capacity=int(rollout_cache_capacity))
        self.cache_min = int(rollout_cache_min_samples)

        # Rollout behavior control (reduces “always bet” hallucinations)
        self.rollout_bet_freq = float(rollout_bet_freq)

        # Debug surfaces for CLI (no interop changes)
        self.last_root_visits: Dict[int, int] = {}
        self.last_root_q: Dict[int, float] = {}

        # In the incomplete-information refactor, hole cards are no longer stored
        # in the public state. The legacy MCTS therefore requires the hero's
        # private hand to be provided at search time.
        self.hero_hand_ids: Optional[Tuple[int, int]] = None

    def search(self, root_state: PublicState, my_hand_ids: Tuple[int, int], iters: int = 2000) -> Tuple[int, Dict[int, float]]:
        self.hero_hand_ids = my_hand_ids
        root = Node(state=root_state)

        for _ in range(int(iters)):
            leaf = self._select_and_expand(root)
            value = self._simulate(leaf.state)  # from leaf player's perspective
            self._backpropagate(leaf, value)

        acts = legal_actions(root_state)
        if not acts:
            raise ValueError("No legal actions at root")

        # Build stats in ROOT player's perspective
        stats: Dict[int, float] = {}
        visits_map: Dict[int, int] = {}

        for a in acts:
            ch = root.children.get(a)
            if ch is None or ch.n == 0:
                stats[a] = 0.0
                visits_map[a] = 0
                continue
            stats[a] = self._child_value_from_parent_perspective(parent=root, child=ch)
            visits_map[a] = ch.n

        # Stash for CLI printing (n, q)
        self.last_root_visits = dict(visits_map)
        self.last_root_q = dict(stats)

        # Choose action by best EV, tie-break by visits
        best_act = max(acts, key=lambda a: (stats.get(a, 0.0), visits_map.get(a, 0)))

        return best_act, stats

    # ---------------- Core MCTS steps ----------------

    def _select_and_expand(self, node: Node) -> Node:
        """
        Selection + one-step expansion.
        Returns a node to simulate from.
        """
        while True:
            acts = legal_actions(node.state)
            if not acts:
                return node  # terminal or no legal actions

            # expand an unvisited action
            unexpanded = [a for a in acts if a not in node.children]
            if unexpanded:
                a = unexpanded[self.rng.randrange(len(unexpanded))]
                child_state = apply_action(node.state, a)
                child = Node(state=child_state, parent=node, parent_action=a)
                node.children[a] = child
                return child

            # all expanded, pick best child by UCB (in this node's perspective)
            node = self._best_child_ucb(node)

    def _best_child_ucb(self, node: Node) -> Node:
        assert node.children, "UCB called with no children"
        logN = math.log(node.n + 1.0)

        def ucb(ch: Node) -> float:
            if ch.n == 0:
                return float("inf")
            mean = self._child_value_from_parent_perspective(parent=node, child=ch)
            return mean + self.c * math.sqrt(logN / ch.n)

        return max(node.children.values(), key=ucb)

    # ---------------- Negamax glue ----------------

    @staticmethod
    def _edge_sign(parent: Node, child: Node) -> float:
        """
        +1 if player-to-act is the same in parent and child, else -1.
        """
        return 1.0 if parent.state.gs.to_act == child.state.gs.to_act else -1.0

    def _child_value_from_parent_perspective(self, parent: Node, child: Node) -> float:
        """
        Child.q is stored from CHILD player's perspective.
        Convert it into PARENT player's perspective.
        """
        return self._edge_sign(parent, child) * child.q()

    # ---------------- Simulation / rollouts ----------------

    def _simulate(self, state: PublicState) -> float:
        """
        Cached rollout by abstraction bucket key.

        IMPORTANT:
        - Cache stores HERO chipEV because bucket_key doesn't include to_act.
        - We convert HERO chipEV -> current player's perspective before returning.
        """
        key = state.gs.bucket_key()
        cached = self.tt.get(key)
        if cached is not None and cached.n >= self.cache_min:
            hero_ev = cached.mean_ev
            return hero_ev if state.gs.to_act == Player.HERO else -hero_ev

        hero_ev = self._rollout_hero_perspective(state)
        self.tt.update(key, hero_ev)

        return hero_ev if state.gs.to_act == Player.HERO else -hero_ev

    def _rollout_hero_perspective(self, state: PublicState) -> float:
        """
        Plays out the hand with:
          - one sampled villain hand
          - one sampled full runout
          - heuristic policy for both players

        Returns HERO-perspective chipEV in CHIPS (not normalized).
        Positive means hero wins chips relative to start-of-hand stack.
        """
        ps = state

        # Sample hidden info once per rollout
        if self.hero_hand_ids is None:
            raise RuntimeError("MCTS.search must be called with my_hand_ids after the private-state refactor")
        vill_hand, full_board = sample_villain_and_runout(ps.gs, self.hero_hand_ids, self.rng)  # full_board padded to 5

        while not is_terminal(ps):
            acts = legal_actions(ps)
            if not acts:
                break

            cur_hand = self.hero_hand_ids if ps.gs.to_act == Player.HERO else vill_hand
            suggested = suggest_rollout_action(ps.gs, cur_hand, rng=self.rng)
            act = suggested.act

            # Clamp + de-bias (reduce “always bet” rollouts)
            act = self._rollout_action_with_bet_throttle(act, acts)

            ps = apply_action(ps, act)

            # Reveal board cards deterministically from sampled runout when street advances
            ps = _reveal_board_for_street(ps, full_board)

        # Fold terminal
        w = terminal_winner(ps)
        if w is not None:
            return self._hero_chip_ev_terminal(ps.gs, hero_wins=(w == Player.HERO))

        # Showdown (or “ran out of actions”)
        cmp = showdown_result(self.hero_hand_ids, vill_hand, full_board)  # +1/0/-1
        if cmp > 0:
            return self._hero_chip_ev_terminal(ps.gs, hero_wins=True)
        if cmp < 0:
            return self._hero_chip_ev_terminal(ps.gs, hero_wins=False)
        return self._hero_chip_ev_terminal(ps.gs, hero_wins=None)

    def _hero_chip_ev_terminal(self, gs: GameState, hero_wins: Optional[bool]) -> float:
        """
        Terminal payout as HERO net chips relative to start-of-hand.

        Convention:
          - Betting actions already moved chips into gs.pot and reduced stacks.
          - At terminal evaluation, we “award” the pot to the winner (or split it).
        """
        meta = gs.meta or {}

        # These should be set in start_postflop_state; fallback keeps code safe.
        hero_start = int(meta.get("hero_start", gs.hero_stack))
        # villain_start unused for hero net EV, but kept for symmetry/debugging
        _vill_start = int(meta.get("villain_start", gs.villain_stack))

        pot = int(gs.pot)
        hero_stack = int(gs.hero_stack)

        if hero_wins is True:
            hero_final = hero_stack + pot
        elif hero_wins is False:
            hero_final = hero_stack
        else:
            # tie: split pot (integer split)
            hero_final = hero_stack + (pot // 2)

        return float(hero_final - hero_start)

    def _rollout_action_with_bet_throttle(self, proposed: int, acts: list[int]) -> int:
        """
        Keep heuristic suggestion but throttle BET frequency, preferring CHECK when possible.
        (This prevents rollouts from teaching the tree that triple-barreling air is always standard.)
        """
        # clamp illegal
        if proposed not in acts:
            proposed = self._fallback_legal_action(acts)

        # only throttle the half-pot BET (not raises / all-ins)
        if proposed == Act.BET:
            if self.rng.random() > self.rollout_bet_freq:
                if Act.CHECK in acts:
                    return Act.CHECK
                if Act.CALL in acts:
                    return Act.CALL
                return self._fallback_legal_action(acts)

        return proposed

    def _fallback_legal_action(self, acts: list[int]) -> int:
        if Act.CHECK in acts:
            return Act.CHECK
        if Act.CALL in acts:
            return Act.CALL
        if Act.FOLD in acts:
            return Act.FOLD
        return acts[0]

    def _backpropagate(self, node: Node, value: float) -> None:
        """
        Negamax backup.
        `value` is from `node` player's perspective.
        Negate when moving across an edge that flips player-to-act.
        """
        cur = node
        v = float(value)

        while cur is not None:
            cur.n += 1
            cur.w += v

            parent = cur.parent
            if parent is None:
                break

            if parent.state.gs.to_act != cur.state.gs.to_act:
                v = -v

            cur = parent


# ---------------- Helpers ----------------

def _reveal_board_for_street(ps: PublicState, full_board_padded: Tuple[int, int, int, int, int]) -> PublicState:
    street = ps.gs.street
    if street <= 0:
        return ps

    target_len = 3 if street == 1 else (4 if street == 2 else 5)

    current = [cid for cid in ps.gs.board if cid != -1]
    if len(current) >= target_len:
        return ps

    desired = [cid for cid in full_board_padded if cid != -1][:target_len]
    return _replace_board_ids(ps, desired)


def _replace_board_ids(ps: PublicState, board_ids_prefix: list[int]) -> PublicState:
    gs = ps.gs
    new_board = pad_board(board_ids_prefix)

    gs2 = GameState(
        hero_hand=gs.hero_hand,
        board=new_board,
        street=gs.street,
        pot=gs.pot,
        hero_stack=gs.hero_stack,
        villain_stack=gs.villain_stack,
        to_act=gs.to_act,
        action_history=gs.action_history,
        meta=gs.meta,
    )

    # NOTE: keep any extra fields in PublicState (like raises_this_street) via getattr default
    # so we don't break if you add it.
    return PublicState(
        gs=gs2,
        street_bet_open=ps.street_bet_open,
        bet_by=ps.bet_by,
        bet_amount=ps.bet_amount,
        **({"raises_this_street": getattr(ps, "raises_this_street")} if hasattr(ps, "raises_this_street") else {}),
    )
