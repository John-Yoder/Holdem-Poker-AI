import pytest

from poker.helpers.abstraction import (
    GameState, Street, Player, Action, Act,
    card_to_id, id_to_card
)

def test_card_roundtrip():
    cid = card_to_id("As")
    c = id_to_card(cid)
    assert str(c) == "As"

def test_action_encode_decode_roundtrip():
    a = Action(street=Street.FLOP, player=Player.HERO, act=Act.BET, amount=123)
    x = a.encode()
    a2 = Action.decode(x)
    assert a2.street == a.street
    assert a2.player == a.player
    assert a2.act == a.act
    assert a2.amount == a.amount

def test_gamestate_bucket_key_stable():
    s = GameState.from_cards(
        hero_hand=["Ah","Qh"],
        board=["Kh","7d","2h"],
        street=Street.FLOP,
        pot=60,
        hero_stack=940,
        villain_stack=940,
        to_act=Player.HERO,
    )
    k1 = s.bucket_key()
    k2 = s.bucket_key()
    assert k1 == k2
    assert len(k1) == 4

def test_apply_action_appends_history():
    s = GameState.from_cards(
        hero_hand=["Ah","Qh"],
        board=["Kh","7d","2h"],
        street=Street.FLOP,
        pot=60,
        hero_stack=940,
        villain_stack=940,
        to_act=Player.HERO,
    )
    s2 = s.apply_action(
        Action(street=Street.FLOP, player=Player.HERO, act=Act.BET, amount=40),
        pot_delta=40,
        hero_stack_delta=-40
    )
    assert len(s.action_history) == 0
    assert len(s2.action_history) == 1
    assert s2.pot == 100
    assert s2.hero_stack == 900
