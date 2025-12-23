# scripts/selfplay_collect.py
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
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

from poker.ai.infoset import bet_bucket_from_amount, infoset_key  # for logging keys
from poker.helpers.abstraction import Player, Act, Street, pad_board, id_to_card


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


# -----------------------------
# Pretty formatting helpers
# -----------------------------
def _fmt_card(cid: int) -> str:
    return "??" if cid == -1 else str(id_to_card(cid))

def _fmt_hand(hand_ids: Tuple[int, int]) -> str:
    return f"{_fmt_card(hand_ids[0])} {_fmt_card(hand_ids[1])}"

def _fmt_board(board_ids: Tuple[int, int, int, int, int]) -> str:
    return " ".join(_fmt_card(c) for c in board_ids if c != -1)


# -----------------------------
# State surgery helpers (same style as your worker)
# -----------------------------
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
    street = int(ps.gs.street)
    if street <= 0:
        return ps
    target_len = 3 if street == int(Street.FLOP) else (4 if street == int(Street.TURN) else 5)
    current = [cid for cid in ps.gs.board if cid != -1]
    if len(current) >= target_len:
        return ps
    desired = [cid for cid in full_board if cid != -1][:target_len]
    return _replace_board_ids(ps, desired)


# -----------------------------
# Sampling helpers (no duplicate cards)
# -----------------------------
def _sample_two_hands_and_flop(rng: random.Random) -> Tuple[List[str], List[str], List[str]]:
    deck = list(range(52))
    rng.shuffle(deck)

    hero = deck[:2]
    vill = deck[2:4]
    flop = deck[4:7]

    hero_str = [str(id_to_card(hero[0])), str(id_to_card(hero[1]))]
    vill_str = [str(id_to_card(vill[0])), str(id_to_card(vill[1]))]
    flop_str = [str(id_to_card(flop[0])), str(id_to_card(flop[1])), str(id_to_card(flop[2]))]
    return hero_str, vill_str, flop_str


# -----------------------------
# Logging helpers (infoset + cache instrumentation)
# -----------------------------
def _facing_bet(ps: PublicState) -> bool:
    return bool(ps.street_bet_open) and int(ps.bet_amount or 0) > 0

def _call_price(ps: PublicState) -> int:
    return int(ps.bet_amount or 0)

def _pot_size(ps: PublicState) -> int:
    return int(ps.gs.pot)

def infoset_key_for_ps(ps: PublicState, my_hand_ids: Tuple[int, int]) -> Any:
    fb = 1 if _facing_bet(ps) else 0
    bb = bet_bucket_from_amount(_pot_size(ps), _call_price(ps)) if fb else 0
    return infoset_key(ps.gs, my_hand_ids, facing_bet=fb, bet_bucket=bb)

def _cache_snapshot(mcts: ISMCTS) -> Dict[str, int]:
    # LRUTranspoTable exposes hits/misses/evictions and __len__ :contentReference[oaicite:1]{index=1}
    tt = mcts.tt
    return {
        "entries": int(len(tt)),
        "hits": int(getattr(tt, "hits", 0)),
        "misses": int(getattr(tt, "misses", 0)),
        "evictions": int(getattr(tt, "evictions", 0)),
    }


# -----------------------------
# Terminal payout (doesn't mutate engine, just computes winner + nets)
# -----------------------------
def _terminal_result(
    ps: PublicState,
    hero_hand: Tuple[int, int],
    vill_hand: Tuple[int, int],
    full_board: Tuple[int, int, int, int, int],
    hero_stack_start: int,
    vill_stack_start: int,
) -> Dict[str, Any]:
    pot = int(ps.gs.pot)
    hero_stack = int(ps.gs.hero_stack)
    vill_stack = int(ps.gs.villain_stack)

    ended_by_fold = False
    winner: Optional[int] = None

    if ps.gs.meta and ps.gs.meta.get("winner", None) is not None:
        winner = int(ps.gs.meta["winner"])
        ended_by_fold = True
    else:
        cmp = showdown_result(hero_hand, vill_hand, full_board)
        if cmp > 0:
            winner = int(Player.HERO)
        elif cmp < 0:
            winner = int(Player.VILLAIN)
        else:
            winner = None

    if winner == int(Player.HERO):
        hero_final = hero_stack + pot
        vill_final = vill_stack
    elif winner == int(Player.VILLAIN):
        hero_final = hero_stack
        vill_final = vill_stack + pot
    else:
        hero_final = hero_stack + pot // 2
        vill_final = vill_stack + (pot - pot // 2)

    return {
        "winner": winner,
        "ended_by_fold": ended_by_fold,
        "hero_stack_final": int(hero_final),
        "vill_stack_final": int(vill_final),
        "hero_net": int(hero_final - hero_stack_start),
        "vill_net": int(vill_final - vill_stack_start),
    }


# -----------------------------
# Main self-play loop
# -----------------------------
def run_selfplay(
    hands: int,
    iters: int,
    seed: int,
    c: float,
    pot_bb: float,
    stacks_bb: float,
    rollout_bet_freq: float,
    disallow_allin: bool,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    mcts = ISMCTS(rng=rng, exploration_c=c, rollout_bet_freq=rollout_bet_freq)

    dataset: Dict[str, Any] = {
        "meta": {
            "created_unix": time.time(),
            "seed": seed,
            "hands": hands,
            "iters_per_decision": iters,
            "exploration_c": c,
            "pot_bb": pot_bb,
            "stacks_bb": stacks_bb,
            "rollout_bet_freq": rollout_bet_freq,
            "disallow_allin": disallow_allin,
            "bb_chips": BB_CHIPS,
        },
        "hands": [],
        "final": {},
    }

    hero_stack = int(round(stacks_bb * BB_CHIPS))
    vill_stack = int(round(stacks_bb * BB_CHIPS))
    pot_chips = int(round(pot_bb * BB_CHIPS))

    hero_is_ip = True  # alternate button each hand

    for hand_i in range(hands):
        if hero_stack <= 0 or vill_stack <= 0:
            break

        # post initial pot (split, villain pays odd chip if any)
        hero_post = pot_chips // 2
        vill_post = pot_chips - hero_post
        hero_stack = max(0, hero_stack - hero_post)
        vill_stack = max(0, vill_stack - vill_post)

        hero_hand_str, vill_hand_str, flop_str = _sample_two_hands_and_flop(rng)

        # engine only needs "hero_hand" + flop to start; we store villain privately for self-play
        stacks_for_engine = max(hero_stack, vill_stack, pot_chips)
        ps, hero_private = start_postflop_state(
            hero_hand=hero_hand_str,
            flop=flop_str,
            pot=pot_chips,
            stacks_bb=stacks_for_engine,
            to_act=int(Player.HERO),
        )

        # overwrite stacks to match our accounting after posting
        gs = ps.gs
        gs2 = gs.__class__(
            board=gs.board,
            street=gs.street,
            pot=pot_chips,
            hero_stack=hero_stack,
            villain_stack=vill_stack,
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

        hero_hand_ids = hero_private.hand  # from engine start
        # make villain ids consistent with strings we sampled (so self-play is truly “both known”)
        # easiest: sample runout with explicit villain ids by temporarily using uniform sampler once:
        vill_hand_ids = sample_opponent_hand_uniform(ps.gs, hero_hand_ids, rng)
        # overwrite villain to match sampled villain_str if you want strict consistency with strings:
        # (If you already have a card_to_id utility exposed, swap it in here.)
        # For now we keep vill_hand_ids uniform-consistent with your belief module.

        full_board_ids = sample_runout_uniform(ps.gs, hero_hand_ids, vill_hand_ids, rng)
        ps = _replace_board_ids(ps, [cid for cid in full_board_ids[:3] if cid != -1])

        oop_player = int(Player.VILLAIN) if hero_is_ip else int(Player.HERO)
        ps = _set_to_act(ps, oop_player)

        hand_rec: Dict[str, Any] = {
            "hand_index": hand_i + 1,
            "button": "HERO" if hero_is_ip else "VILLAIN",
            "posted": {"hero": hero_post, "villain": vill_post, "pot": pot_chips},
            "stacks_start_after_post": {"hero": hero_stack, "villain": vill_stack},
            "setup": {
                "hero_hand": _fmt_hand(hero_hand_ids),
                "vill_hand": _fmt_hand(vill_hand_ids),
                "flop": " ".join(flop_str),
                "full_board": _fmt_board(full_board_ids),
            },
            "actions": [],
            "terminal": {},
        }

        hero_stack_start = hero_stack
        vill_stack_start = vill_stack

        safety = 0
        while True:
            ps = _reveal_board_for_street(ps, full_board_ids)

            if is_terminal(ps) or (ps.gs.meta and ps.gs.meta.get("terminal", False)):
                break

            acts = [int(a) for a in legal_actions(ps)]
            if disallow_allin:
                acts = [a for a in acts if a != int(Act.ALLIN)]
            if not acts:
                break

            actor = int(ps.gs.to_act)
            my_hand = hero_hand_ids if actor == int(Player.HERO) else vill_hand_ids

            cache_before = _cache_snapshot(mcts)

            best_act, q_map = mcts.search(ps, my_hand, iters=iters)
            best_act = int(best_act)

            # if ALLIN is filtered, downgrade defensively
            if disallow_allin and best_act == int(Act.ALLIN):
                if int(Act.RAISE) in acts:
                    best_act = int(Act.RAISE)
                elif int(Act.BET) in acts:
                    best_act = int(Act.BET)
                elif int(Act.CALL) in acts:
                    best_act = int(Act.CALL)
                elif int(Act.CHECK) in acts:
                    best_act = int(Act.CHECK)
                else:
                    best_act = acts[0]

            cache_after = _cache_snapshot(mcts)

            I = infoset_key_for_ps(ps, my_hand)
            node = mcts.table.get(I)

            step = {
                "street": int(ps.gs.street),
                "street_name": STREET_NAME.get(int(ps.gs.street), str(int(ps.gs.street))),
                "board": _fmt_board(ps.gs.board),
                "pot": int(ps.gs.pot),
                "stacks": {"hero": int(ps.gs.hero_stack), "villain": int(ps.gs.villain_stack)},
                "to_act": actor,
                "to_act_name": "HERO" if actor == int(Player.HERO) else "VILLAIN",
                "street_bet_open": bool(ps.street_bet_open),
                "bet_by": int(ps.bet_by),
                "bet_amount": int(ps.bet_amount or 0),
                "legal_actions": acts,
                "legal_action_names": [ACT_NAME.get(a, str(a)) for a in acts],

                "infoset_key_repr": repr(I),
                "chosen_act": best_act,
                "chosen_act_name": ACT_NAME.get(best_act, str(best_act)),

                # root stats the search stored
                "root_visits": dict(getattr(mcts, "last_root_visits", {})),
                "root_q": dict(getattr(mcts, "last_root_q", {})),

                # infoset node stats (if present)
                "infoset_node": None if node is None else {
                    "n": int(getattr(node, "n", 0)),
                    "na": {int(k): int(v) for k, v in getattr(node, "na", {}).items()},
                    "wa": {int(k): float(v) for k, v in getattr(node, "wa", {}).items()},
                },

                # rollout cache instrumentation (deltas help per-decision analysis)
                "cache": {
                    "before": cache_before,
                    "after": cache_after,
                    "delta": {
                        "hits": cache_after["hits"] - cache_before["hits"],
                        "misses": cache_after["misses"] - cache_before["misses"],
                        "evictions": cache_after["evictions"] - cache_before["evictions"],
                        "entries": cache_after["entries"],
                    },
                },
            }

            hand_rec["actions"].append(step)

            ps = apply_action(ps, best_act)
            safety += 1
            if safety > 100:
                # if this trips, your game rules are probably not advancing street or terminaling properly
                hand_rec["terminal"] = {"error": "Safety stop: too many actions"}
                break

        # terminal bookkeeping
        ps = _reveal_board_for_street(ps, full_board_ids)
        term = _terminal_result(ps, hero_hand_ids, vill_hand_ids, full_board_ids, hero_stack_start, vill_stack_start)

        hero_stack = int(term["hero_stack_final"])
        vill_stack = int(term["vill_stack_final"])

        hand_rec["terminal"] = {
            "winner": term["winner"],
            "winner_name": ("HERO" if term["winner"] == int(Player.HERO) else ("VILLAIN" if term["winner"] == int(Player.VILLAIN) else "TIE")),
            "ended_by_fold": bool(term["ended_by_fold"]),
            "hero_net": int(term["hero_net"]),
            "vill_net": int(term["vill_net"]),
            "stacks_final": {"hero": hero_stack, "villain": vill_stack},
            "final_board": _fmt_board(full_board_ids),
        }

        dataset["hands"].append(hand_rec)

        # alternate IP/OOP each hand
        hero_is_ip = not hero_is_ip

    dataset["final"] = {
        "hands_played": len(dataset["hands"]),
        "hero_stack": hero_stack,
        "vill_stack": vill_stack,
        "infosets_total": len(getattr(mcts, "table", {})),
        "rollout_cache": _cache_snapshot(mcts),
    }
    return dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, default=200)
    ap.add_argument("--iters", type=int, default=800)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--c", type=float, default=1.4)
    ap.add_argument("--pot_bb", type=float, default=6.0)
    ap.add_argument("--stacks_bb", type=float, default=150.0)
    ap.add_argument("--rollout_bet_freq", type=float, default=0.30)
    ap.add_argument("--no_allin", action="store_true")
    ap.add_argument("--out", type=str, default="selfplay_dataset.json")
    args = ap.parse_args()

    data = run_selfplay(
        hands=args.hands,
        iters=args.iters,
        seed=args.seed,
        c=args.c,
        pot_bb=args.pot_bb,
        stacks_bb=args.stacks_bb,
        rollout_bet_freq=args.rollout_bet_freq,
        disallow_allin=bool(args.no_allin),
    )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Wrote {args.out} with {data['final']['hands_played']} hands.")
    print("Final:", data["final"])


if __name__ == "__main__":
    main()
