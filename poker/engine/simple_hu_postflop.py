from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from ..helpers.abstraction import GameState, Street, Player, Act, Action, id_to_card, card_to_id, pad_board, PrivateInfo
from ..helpers.cards import make_deck
from ..helpers.evaluator import compare_hands


@dataclass(frozen=True, slots=True)
class PublicState:
    """
    Public betting state (no hidden villain cards here).
    We keep this minimal and deterministic. Hidden info is sampled in rollouts.
    """
    gs: GameState
    street_bet_open: bool          # has someone bet this street?
    bet_by: int                    # Player enum id who bet; undefined if street_bet_open=False
    bet_amount: int                # chips amount to call; 0 if none
    raises_this_street: int = 0    # NEW: allow at most 1 raise per street


def start_postflop_state(
    hero_hand: Sequence[str],
    flop: Sequence[str],
    pot: int = 3,
    stacks_bb: int = 100,
    to_act: int = Player.HERO,
) -> Tuple[PublicState, PrivateInfo]:
    """
    Creates a postflop starting point.

    IMPORTANT MODELING CONVENTION:
    The pot represents chips already invested preflop, so those chips must be
    removed from stacks. Otherwise folding postflop is “free” (EV=0) and the
    agent will make absurd folds with strong hands.
    """
    # Split the existing pot between players as “already invested”.
    # If pot is odd, give the extra chip to villain (arbitrary but consistent).
    hero_invest = pot // 2
    villain_invest = pot - hero_invest

    hero_stack0 = stacks_bb - hero_invest
    villain_stack0 = stacks_bb - villain_invest
    if hero_stack0 < 0 or villain_stack0 < 0:
        raise ValueError("Pot is larger than available stacks_bb in start_postflop_state")

    meta = {
        "hero_start": stacks_bb,
        "villain_start": stacks_bb,
        "hero_invest": hero_invest,
        "villain_invest": villain_invest,
    }

    hero_private = PrivateInfo.from_cards(hero_hand)
    gs = GameState.from_cards(
        board=flop,
        street=Street.FLOP,
        pot=pot,
        hero_stack=hero_stack0,
        villain_stack=villain_stack0,
        to_act=to_act,
        action_history=(),
        meta=meta,
    )
    return PublicState(gs=gs, street_bet_open=False, bet_by=Player.HERO, bet_amount=0, raises_this_street=0), hero_private


def legal_actions(ps: PublicState) -> List[int]:
    gs = ps.gs
    if is_terminal(ps):
        return []

    if not ps.street_bet_open:
        # you can always check, bet half-pot, or jam
        return [Act.CHECK, Act.BET, Act.ALLIN]

    # facing a bet
    acts = [Act.CALL, Act.FOLD, Act.ALLIN]
    # allow one raise per street
    if ps.raises_this_street == 0:
        acts.append(Act.RAISE)
    return acts


def half_pot_bet_size(pot: int) -> int:
    return max(1, pot // 2)


def apply_action(ps: PublicState, act: int) -> PublicState:
    gs = ps.gs
    p = gs.to_act

    if act not in legal_actions(ps):
        raise ValueError(f"Illegal action {act} for state")

    def swap_player(x: int) -> int:
        return Player.VILLAIN if x == Player.HERO else Player.HERO

    def stack_of(player: int) -> int:
        return gs.hero_stack if player == Player.HERO else gs.villain_stack

    def _apply_put_in(gs0: GameState, player: int, amount: int, a: int) -> GameState:
        # put amount into pot immediately (same convention as your BET/CALL)
        pot_delta = amount
        hero_delta = -amount if player == Player.HERO else 0
        vill_delta = -amount if player == Player.VILLAIN else 0
        return gs0.apply_action(
            Action(street=gs0.street, player=player, act=a, amount=amount),
            pot_delta=pot_delta,
            hero_stack_delta=hero_delta,
            villain_stack_delta=vill_delta,
        )

    def _spr_after_call(pot_now: int, hero_stack_now: int, vill_stack_now: int, add_agg: int, add_call: int) -> float:
        pot_after = pot_now + add_agg + add_call
        hero_after = hero_stack_now
        vill_after = vill_stack_now
        # the caller is always the other player
        # we’ll subtract from the correct stacks outside (more explicit below)
        eff = min(hero_after, vill_after)
        return float("inf") if pot_after <= 0 else eff / pot_after

    def _should_shove_after_called(
        pot_now: int,
        hero_stack_now: int,
        vill_stack_now: int,
        aggressor: int,
        agg_put_in: int,
        caller_put_in: int,
    ) -> bool:
        # compute stacks after both put money in
        hero_after = hero_stack_now - (agg_put_in if aggressor == Player.HERO else caller_put_in)
        vill_after = vill_stack_now - (agg_put_in if aggressor == Player.VILLAIN else caller_put_in)
        pot_after = pot_now + agg_put_in + caller_put_in
        eff = min(hero_after, vill_after)
        spr = float("inf") if pot_after <= 0 else eff / pot_after
        return spr < 0.5

    # ---------------- no bet yet this street ----------------
    if not ps.street_bet_open:
        if act == Act.CHECK:
            gs2 = gs.apply_action(Action(street=gs.street, player=p, act=Act.CHECK, amount=0))

            hist = gs2.action_history
            advanced = False
            if len(hist) >= 2:
                a1 = Action.decode(hist[-1])
                a2 = Action.decode(hist[-2])
                if a1.act == Act.CHECK and a2.act == Act.CHECK and a1.street == a2.street == gs.street:
                    advanced = True

            if advanced:
                return PublicState(
                    gs=_advance_street(gs2, next_to_act=Player.HERO),
                    street_bet_open=False,
                    bet_by=Player.HERO,
                    bet_amount=0,
                    raises_this_street=0,
                )
            return PublicState(
                gs=_replace_to_act(gs2, swap_player(p)),
                street_bet_open=False,
                bet_by=Player.HERO,
                bet_amount=0,
                raises_this_street=0,
            )

        # BET half pot or ALLIN (ALLIN explicitly, or auto-upgrade below)
        if act in (Act.BET, Act.ALLIN):
            if act == Act.BET:
                size = half_pot_bet_size(gs.pot)
            else:
                size = stack_of(p)

            size = min(size, stack_of(p))

            # AUTO-ALLIN rule: if bet+call makes SPR < 0.5, shove instead
            caller = swap_player(p)
            call_amt = min(size, stack_of(caller))
            if act == Act.BET and _should_shove_after_called(gs.pot, gs.hero_stack, gs.villain_stack, p, size, call_amt):
                size = stack_of(p)  # shove full stack
                act = Act.ALLIN
                call_amt = min(size, stack_of(caller))

            gs2 = _apply_put_in(gs, p, size, act)

            return PublicState(
                gs=_replace_to_act(gs2, swap_player(p)),
                street_bet_open=True,
                bet_by=p,
                bet_amount=size,          # amount the other player must call
                raises_this_street=0,
            )

    # ---------------- facing a bet ----------------
    else:
        # facing amount to call is ps.bet_amount
        to_call = ps.bet_amount

        if act == Act.FOLD:
            gs2 = gs.apply_action(Action(street=gs.street, player=p, act=Act.FOLD, amount=0))
            return PublicState(
                gs=_with_terminal(gs2, winner=swap_player(p)),
                street_bet_open=True,
                bet_by=ps.bet_by,
                bet_amount=ps.bet_amount,
                raises_this_street=ps.raises_this_street,
            )

        if act == Act.CALL:
            call_amt = min(to_call, stack_of(p))
            gs2 = _apply_put_in(gs, p, call_amt, Act.CALL)

            return PublicState(
                gs=_advance_street(gs2, next_to_act=Player.HERO),
                street_bet_open=False,
                bet_by=Player.HERO,
                bet_amount=0,
                raises_this_street=0,
            )

        if act == Act.ALLIN:
            # all-in as a raise over the current bet (if possible),
            # otherwise it degenerates to CALL when you’re short.
            allin_amt = stack_of(p)
            if allin_amt <= to_call:
                # can’t raise, just call (all-in call)
                gs2 = _apply_put_in(gs, p, allin_amt, Act.CALL)
                return PublicState(
                    gs=_advance_street(gs2, next_to_act=Player.HERO),
                    street_bet_open=False,
                    bet_by=Player.HERO,
                    bet_amount=0,
                    raises_this_street=0,
                )

            # you put in your full stack now
            gs2 = _apply_put_in(gs, p, allin_amt, Act.ALLIN)

            # now the original bettor must call the difference
            caller = swap_player(p)
            prev_invest = to_call  # what caller already put in when they bet
            new_to_call = allin_amt - prev_invest

            return PublicState(
                gs=_replace_to_act(gs2, caller),
                street_bet_open=True,
                bet_by=p,
                bet_amount=new_to_call,
                raises_this_street=1,
            )

        if act == Act.RAISE:
            if ps.raises_this_street != 0:
                raise ValueError("Re-raises not allowed in this toy model")

            # raise-to = 3x the current bet amount (total amount raiser puts in)
            raise_to = 3 * to_call
            raise_to = min(raise_to, stack_of(p))

            # AUTO-ALLIN rule: if raise+call makes SPR < 0.5, shove instead
            # Caller (original bettor) has already invested `to_call`
            caller = swap_player(p)
            caller_extra = min(max(0, raise_to - to_call), stack_of(caller))
            if _should_shove_after_called(gs.pot, gs.hero_stack, gs.villain_stack, p, raise_to, caller_extra):
                raise_to = stack_of(p)  # shove full stack
                # if shove <= to_call, it’s just a call (but this case is rare)
                if raise_to <= to_call:
                    call_amt = min(to_call, stack_of(p))
                    gs2 = _apply_put_in(gs, p, call_amt, Act.CALL)
                    return PublicState(
                        gs=_advance_street(gs2, next_to_act=Player.HERO),
                        street_bet_open=False,
                        bet_by=Player.HERO,
                        bet_amount=0,
                        raises_this_street=0,
                    )
                gs2 = _apply_put_in(gs, p, raise_to, Act.ALLIN)
                new_to_call = raise_to - to_call
                return PublicState(
                    gs=_replace_to_act(gs2, caller),
                    street_bet_open=True,
                    bet_by=p,
                    bet_amount=new_to_call,
                    raises_this_street=1,
                )

            # normal raise
            gs2 = _apply_put_in(gs, p, raise_to, Act.RAISE)
            new_to_call = raise_to - to_call  # bettor already has to_call in the pot
            return PublicState(
                gs=_replace_to_act(gs2, swap_player(p)),
                street_bet_open=True,
                bet_by=p,
                bet_amount=new_to_call,
                raises_this_street=1,
            )

    raise RuntimeError("Unhandled transition")


def is_terminal(ps: PublicState) -> bool:
    return bool(ps.gs.meta and ps.gs.meta.get("terminal", False))


def terminal_winner(ps: PublicState) -> Optional[int]:
    if not is_terminal(ps):
        return None
    if not ps.gs.meta:
        return None
    w = ps.gs.meta.get("winner", None)
    return None if w is None else int(w)


def _replace_to_act(gs: GameState, to_act: int) -> GameState:
    return GameState(
        board=gs.board,
        street=gs.street,
        pot=gs.pot,
        hero_stack=gs.hero_stack,
        villain_stack=gs.villain_stack,
        to_act=to_act,
        action_history=gs.action_history,
        meta=gs.meta,
    )


def _with_terminal(gs: GameState, winner: int) -> GameState:
    meta = dict(gs.meta) if gs.meta else {}
    meta["terminal"] = True
    meta["winner"] = winner
    return GameState(
        board=gs.board,
        street=gs.street,
        pot=gs.pot,
        hero_stack=gs.hero_stack,
        villain_stack=gs.villain_stack,
        to_act=gs.to_act,
        action_history=gs.action_history,
        meta=meta,
    )


def _advance_street(gs: GameState, next_to_act: int) -> GameState:
    if gs.street >= Street.RIVER:
        meta = dict(gs.meta) if gs.meta else {}
        meta["showdown"] = True
        meta["terminal"] = True
        return GameState(
            board=gs.board,
            street=gs.street,
            pot=gs.pot,
            hero_stack=gs.hero_stack,
            villain_stack=gs.villain_stack,
            to_act=next_to_act,
            action_history=gs.action_history,
            meta=meta,
        )

    return GameState(
        board=gs.board,
        street=gs.street + 1,
        pot=gs.pot,
        hero_stack=gs.hero_stack,
        villain_stack=gs.villain_stack,
        to_act=next_to_act,
        action_history=gs.action_history,
        meta=gs.meta,
    )


def _remaining_deck_ids(exclude_ids: Sequence[int]) -> List[int]:
    excl_cards = [id_to_card(x) for x in exclude_ids]
    return [card_to_id(str(c)) for c in make_deck(exclude=excl_cards)]

def showdown_result(hero_hand_ids: Tuple[int, int], vill_hand_ids: Tuple[int, int], board_ids: Tuple[int, int, int, int, int]) -> int:
    hero = [id_to_card(hero_hand_ids[0]), id_to_card(hero_hand_ids[1])]
    vill = [id_to_card(vill_hand_ids[0]), id_to_card(vill_hand_ids[1])]
    board = [id_to_card(x) for x in board_ids]
    cmp = compare_hands(hero, vill, board)
    return 1 if cmp > 0 else (-1 if cmp < 0 else 0)
