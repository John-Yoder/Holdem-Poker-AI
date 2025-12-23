from __future__ import annotations

import json
import sys
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from poker.engine.simple_hu_postflop import (
    PublicState,
    start_postflop_state,
    legal_actions,
    apply_action,
    is_terminal,
    showdown_result,
)
from poker.ai.ismcts import ISMCTS
from poker.ai.belief import sample_opponent_hand_uniform, sample_runout_uniform
from poker.helpers.abstraction import Player, Act, Street, pad_board, id_to_card
from poker.helpers.evaluator import evaluate_best


BB_CHIPS = 100

ACT_NAME = {
    int(Act.FOLD): "FOLD",
    int(Act.CHECK): "CHECK",
    int(Act.CALL): "CALL",
    int(Act.BET): "BET_HALF_POT",
    int(Act.RAISE): "RAISE_3X",
    int(Act.ALLIN): "ALLIN",
}

STREET_NAME = {
    int(Street.PREFLOP): "PREFLOP",
    int(Street.FLOP): "FLOP",
    int(Street.TURN): "TURN",
    int(Street.RIVER): "RIVER",
}


def _jwrite(obj: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _fmt_card(cid: int) -> str:
    return "??" if cid == -1 else str(id_to_card(cid))


def _fmt_hand(hand_ids: Tuple[int, int]) -> str:
    return f"{_fmt_card(hand_ids[0])} {_fmt_card(hand_ids[1])}"


def _fmt_board(board_ids: Tuple[int, int, int, int, int]) -> str:
    return " ".join(_fmt_card(c) for c in board_ids if c != -1)


def _hand_strength_name(hand_ids: Tuple[int, int], board_ids: Tuple[int, int, int, int, int]) -> str:
    board = [str(id_to_card(cid)) for cid in board_ids if cid != -1]
    if len(board) < 3:
        return "unknown"
    hand = [str(id_to_card(hand_ids[0])), str(id_to_card(hand_ids[1]))]
    name, _, _ = evaluate_best(hand, board)
    return str(name).replace("_", " ")


def _replace_board_ids(ps: PublicState, board_ids_prefix: List[int]) -> PublicState:
    new_board = pad_board(board_ids_prefix)
    gs = ps.gs
    gs2 = gs.__class__(
        board=new_board,
        street=gs.street,
        pot=gs.pot,
        hero_stack=gs.hero_stack,
        villain_stack=gs.villain_stack,
        to_act=gs.to_act,
        action_history=gs.action_history,
        meta=gs.meta,
    )
    kwargs = {}
    if hasattr(ps, "raises_this_street"):
        kwargs["raises_this_street"] = getattr(ps, "raises_this_street")
    return ps.__class__(
        gs=gs2,
        street_bet_open=ps.street_bet_open,
        bet_by=ps.bet_by,
        bet_amount=ps.bet_amount,
        **kwargs,
    )


def _set_to_act(ps: PublicState, to_act: int) -> PublicState:
    gs = ps.gs
    gs2 = gs.__class__(
        board=gs.board,
        street=gs.street,
        pot=gs.pot,
        hero_stack=gs.hero_stack,
        villain_stack=gs.villain_stack,
        to_act=int(to_act),
        action_history=gs.action_history,
        meta=gs.meta,
    )
    kwargs = {}
    if hasattr(ps, "raises_this_street"):
        kwargs["raises_this_street"] = getattr(ps, "raises_this_street")
    return ps.__class__(
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


def _sample_random_hand_and_flop(rng: random.Random) -> Tuple[List[str], List[str]]:
    deck = list(range(52))
    rng.shuffle(deck)
    hero = deck[:2]
    flop = deck[2:5]
    hero_str = [str(id_to_card(hero[0])), str(id_to_card(hero[1]))]
    flop_str = [str(id_to_card(flop[0])), str(id_to_card(flop[1])), str(id_to_card(flop[2]))]
    return hero_str, flop_str


def _award_pot(
    ps: PublicState,
    hero_hand: Tuple[int, int],
    vill_hand: Tuple[int, int],
    full_board: Tuple[int, int, int, int, int],
    hero_stack_start: int,
    vill_stack_start: int,
) -> Dict[str, Any]:
    gs = ps.gs
    pot = int(gs.pot)

    hero_stack = int(gs.hero_stack)
    vill_stack = int(gs.villain_stack)

    winner: Optional[int] = None
    ended_by_fold = False

    if gs.meta and gs.meta.get("winner", None) is not None:
        winner = int(gs.meta["winner"])
        ended_by_fold = True
    else:
        cmp = showdown_result(hero_hand, vill_hand, full_board)
        if cmp > 0:
            winner = Player.HERO
        elif cmp < 0:
            winner = Player.VILLAIN
        else:
            winner = None

    if winner == Player.HERO:
        hero_stack += pot
    elif winner == Player.VILLAIN:
        vill_stack += pot
    else:
        hero_stack += pot // 2
        vill_stack += pot - (pot // 2)

    return {
        "winner": winner,
        "ended_by_fold": ended_by_fold,
        "hero_stack_final": hero_stack,
        "vill_stack_final": vill_stack,
        "hero_net": hero_stack - hero_stack_start,
        "vill_net": vill_stack - vill_stack_start,
    }


@dataclass
class MatchState:
    rng: random.Random
    mcts: ISMCTS
    hands_total: int
    hands_done: int
    pot_chips: int
    disallow_allin: bool
    human_is: int
    hero_is_ip: bool

    hero_stack: int
    vill_stack: int

    ps: Optional[PublicState] = None
    hero_hand_ids: Optional[Tuple[int, int]] = None
    vill_hand_ids: Optional[Tuple[int, int]] = None
    full_board_ids: Optional[Tuple[int, int, int, int, int]] = None
    oop_player: Optional[int] = None
    ip_player: Optional[int] = None
    last_street_seen: Optional[int] = None


STATE: Optional[MatchState] = None


def _legal_actions_filtered(ps: PublicState, disallow_allin: bool) -> List[int]:
    acts = list(legal_actions(ps))
    if disallow_allin:
        acts = [a for a in acts if a != Act.ALLIN]
    return [int(a) for a in acts]


def _human_str(st: MatchState) -> str:
    return "hero" if int(st.human_is) == int(Player.HERO) else "villain"


def _start_new_hand(st: MatchState) -> None:
    hero_hand_str, flop_str = _sample_random_hand_and_flop(st.rng)

    # --- TAKE THE "INITIAL POT" OUT OF STACKS (split evenly) ---
    pot = int(st.pot_chips)

    # split pot as evenly as possible; if odd, villain pays the extra chip
    hero_post = pot // 2
    vill_post = pot - hero_post

    # don't allow negative stacks
    st.hero_stack = max(0, int(st.hero_stack) - hero_post)
    st.vill_stack = max(0, int(st.vill_stack) - vill_post)

    # engine expects stacks_bb (your engine usage is really "chips"); keep your convention:
    stacks_for_engine = max(st.hero_stack, st.vill_stack, pot)

    ps, hero_private = start_postflop_state(
        hero_hand=hero_hand_str,
        flop=flop_str,
        pot=pot,                    # pot exists because we just posted it
        stacks_bb=stacks_for_engine,
        to_act=Player.HERO,
    )

    # overwrite stacks to match match accounting (after posting)
    gs = ps.gs
    gs2 = gs.__class__(
        board=gs.board,
        street=gs.street,
        pot=pot,
        hero_stack=st.hero_stack,
        villain_stack=st.vill_stack,
        to_act=gs.to_act,
        action_history=gs.action_history,
        meta=gs.meta,
    )
    ps = ps.__class__(
        gs=gs2,
        street_bet_open=ps.street_bet_open,
        bet_by=ps.bet_by,
        bet_amount=ps.bet_amount,
        **({"raises_this_street": getattr(ps, "raises_this_street")} if hasattr(ps, "raises_this_street") else {}),
    )

    hero_hand_ids = hero_private.hand
    vill_hand_ids = sample_opponent_hand_uniform(ps.gs, hero_hand_ids, st.rng)
    full_board_ids = sample_runout_uniform(ps.gs, hero_hand_ids, vill_hand_ids, st.rng)

    # lock visible board to flop initially
    ps = _replace_board_ids(ps, [cid for cid in full_board_ids[:3] if cid != -1])

    st.oop_player = Player.VILLAIN if st.hero_is_ip else Player.HERO
    st.ip_player = Player.HERO if st.hero_is_ip else Player.VILLAIN
    st.last_street_seen = int(ps.gs.street)

    # OOP acts first each street
    ps = _set_to_act(ps, int(st.oop_player))

    st.ps = ps
    st.hero_hand_ids = hero_hand_ids
    st.vill_hand_ids = vill_hand_ids
    st.full_board_ids = full_board_ids

    your_ids = st.hero_hand_ids if int(st.human_is) == int(Player.HERO) else st.vill_hand_ids
    your_strength = _hand_strength_name(your_ids, ps.gs.board)

    _jwrite(
        {
            "type": "hand_start",
            "hand_num": st.hands_done + 1,
            "human": _human_str(st),
            "hero_is_ip": bool(st.hero_is_ip),

            # stacks already reflect posting
            "hero_stack": int(st.hero_stack),
            "vill_stack": int(st.vill_stack),

            "flop": _fmt_board(ps.gs.board),
            "your_hand": _fmt_hand(your_ids),
            "your_strength": your_strength,
        }
    )



def _maybe_street_reset_to_oop(st: MatchState) -> None:
    assert st.ps is not None
    ps = st.ps
    cur_street = int(ps.gs.street)
    if st.last_street_seen is None or cur_street != st.last_street_seen:
        st.last_street_seen = cur_street
        st.ps = _set_to_act(ps, int(st.oop_player))


def _emit_state(st: MatchState) -> None:
    assert st.ps is not None
    assert st.hero_hand_ids is not None
    assert st.vill_hand_ids is not None

    ps = st.ps

    your_ids = st.hero_hand_ids if int(st.human_is) == int(Player.HERO) else st.vill_hand_ids
    your_strength = _hand_strength_name(your_ids, ps.gs.board)
    street_i = int(ps.gs.street)

    _jwrite(
        {
            "type": "state",
            "human": _human_str(st),
            "street": street_i,
            "street_name": STREET_NAME.get(street_i, str(street_i)),
            "board": _fmt_board(ps.gs.board),
            "pot": int(ps.gs.pot),
            "hero_stack": int(ps.gs.hero_stack),
            "vill_stack": int(ps.gs.villain_stack),
            "to_act": int(ps.gs.to_act),
            "street_bet_open": bool(ps.street_bet_open),
            "bet_amount": int(ps.bet_amount or 0),
            "bet_by": int(ps.bet_by),
            "legal_actions": _legal_actions_filtered(ps, st.disallow_allin),

            # ALWAYS show both hands (training mode)
            "hero_hand": _fmt_hand(st.hero_hand_ids),
            "vill_hand": _fmt_hand(st.vill_hand_ids),

            "your_strength": your_strength,
        }
    )


def _step_until_human_needed(st: MatchState, iters: int) -> None:
    assert st.ps is not None
    assert st.full_board_ids is not None
    assert st.hero_hand_ids is not None
    assert st.vill_hand_ids is not None

    safety = 0
    while True:
        st.ps = _reveal_board_for_street(st.ps, st.full_board_ids)
        _maybe_street_reset_to_oop(st)

        ps = st.ps
        if is_terminal(ps) or (ps.gs.meta and ps.gs.meta.get("terminal", False)):
            break

        acts = _legal_actions_filtered(ps, st.disallow_allin)
        if not acts:
            break

        if int(ps.gs.to_act) == int(st.human_is):
            _emit_state(st)
            return

        actor = int(ps.gs.to_act)
        my_hand = st.hero_hand_ids if actor == Player.HERO else st.vill_hand_ids
        best_act, _ = st.mcts.search(ps, my_hand, iters=iters)

        a = int(best_act)
        if st.disallow_allin and a == int(Act.ALLIN):
            if int(Act.RAISE) in acts:
                a = int(Act.RAISE)
            elif int(Act.BET) in acts:
                a = int(Act.BET)
            elif int(Act.CALL) in acts:
                a = int(Act.CALL)
            elif int(Act.CHECK) in acts:
                a = int(Act.CHECK)
            else:
                a = acts[0]

        st.ps = apply_action(ps, a)
        _jwrite({"type": "action", "actor": actor, "act": a, "act_name": ACT_NAME.get(a, str(a))})

        safety += 1
        if safety > 80:
            _jwrite({"type": "error", "message": "Safety stop: too many actions."})
            break

    ps = st.ps
    result = _award_pot(
        ps=ps,
        hero_hand=st.hero_hand_ids,
        vill_hand=st.vill_hand_ids,
        full_board=st.full_board_ids,
        hero_stack_start=st.hero_stack,
        vill_stack_start=st.vill_stack,
    )
    st.hero_stack = int(result["hero_stack_final"])
    st.vill_stack = int(result["vill_stack_final"])

    ended_by_fold = bool(result.get("ended_by_fold", False))

    _jwrite(
        {
            "type": "hand_end",
            "hand_num": st.hands_done + 1,
            "final_board": _fmt_board(st.full_board_ids),

            # ALWAYS show both hands, even if ended_by_fold
            "hero_hand": _fmt_hand(st.hero_hand_ids),
            "vill_hand": _fmt_hand(st.vill_hand_ids),

            "winner": result["winner"],
            "hero_net": result["hero_net"],
            "vill_net": result["vill_net"],
            "hero_stack": st.hero_stack,
            "vill_stack": st.vill_stack,
            "ended_by_fold": ended_by_fold,
        }
    )

    st.hands_done += 1
    st.hero_is_ip = not st.hero_is_ip

    st.ps = None
    st.hero_hand_ids = None
    st.vill_hand_ids = None
    st.full_board_ids = None
    st.oop_player = None
    st.ip_player = None
    st.last_street_seen = None

    if st.hands_done < st.hands_total and st.hero_stack > 0 and st.vill_stack > 0:
        _jwrite({"type": "await_next_hand"})
    else:
        _jwrite({"type": "match_end", "hero_stack": st.hero_stack, "vill_stack": st.vill_stack})


def handle(msg: Dict[str, Any]) -> None:
    global STATE
    t = msg.get("type")

    if t == "start_match":
        seed = int(msg.get("seed", 42))
        hands = int(msg.get("hands", 10))
        iters = int(msg.get("iters", 800))
        c = float(msg.get("c", 1.4))
        pot_bb = float(msg.get("pot_bb", 6.0))
        stacks_bb = float(msg.get("stacks_bb", 150.0))
        disallow_allin = bool(msg.get("no_allin", False))
        human = msg.get("human", "hero")
        start_ip = msg.get("start_ip", "hero")
        rollout_bet_freq = float(msg.get("rollout_bet_freq", 0.55))

        rng = random.Random(seed)
        mcts = ISMCTS(rng=rng, exploration_c=c, rollout_bet_freq=rollout_bet_freq)

        human_is = Player.HERO if human == "hero" else Player.VILLAIN
        hero_is_ip = True if start_ip == "hero" else False

        STATE = MatchState(
            rng=rng,
            mcts=mcts,
            hands_total=hands,
            hands_done=0,
            pot_chips=int(round(pot_bb * BB_CHIPS)),
            disallow_allin=disallow_allin,
            human_is=int(human_is),
            hero_is_ip=hero_is_ip,
            hero_stack=int(round(stacks_bb * BB_CHIPS)),
            vill_stack=int(round(stacks_bb * BB_CHIPS)),
        )

        _jwrite({"type": "match_started", "human": human, "start_ip": start_ip})
        _start_new_hand(STATE)
        _step_until_human_needed(STATE, iters=iters)
        return

    if STATE is None:
        _jwrite({"type": "error", "message": "No match. Send start_match first."})
        return

    st = STATE

    if t == "human_action":
        iters = int(msg.get("iters", 800))
        act = int(msg.get("act"))

        if st.ps is None:
            _jwrite({"type": "error", "message": "No active hand."})
            return

        ps = st.ps
        legal = _legal_actions_filtered(ps, st.disallow_allin)
        if act not in legal:
            _jwrite({"type": "error", "message": f"Illegal action {act}. Legal={legal}"})
            _emit_state(st)
            return

        st.ps = apply_action(ps, act)
        _jwrite({"type": "action", "actor": int(ps.gs.to_act), "act": act, "act_name": ACT_NAME.get(act, str(act))})
        _step_until_human_needed(st, iters=iters)
        return

    if t == "next_hand":
        iters = int(msg.get("iters", 800))
        if st.ps is not None:
            _jwrite({"type": "error", "message": "Hand still active; cannot start next hand."})
            return

        if st.hands_done >= st.hands_total or st.hero_stack <= 0 or st.vill_stack <= 0:
            _jwrite({"type": "match_end", "hero_stack": st.hero_stack, "vill_stack": st.vill_stack})
            return

        _start_new_hand(st)
        _step_until_human_needed(st, iters=iters)
        return

    _jwrite({"type": "error", "message": f"Unknown message type: {t}"})


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
            handle(msg)
        except Exception as e:
            _jwrite({"type": "error", "message": f"Worker exception: {e}"})


if __name__ == "__main__":
    main()
