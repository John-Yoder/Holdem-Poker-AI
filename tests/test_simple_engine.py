from poker.engine.simple_hu_postflop import start_postflop_state, legal_actions, apply_action, is_terminal
from poker.helpers.abstraction import Act, Street, Player

def test_engine_legal_actions_start():
    ps = start_postflop_state(["Ah","Qh"], ["Kh","7d","2h"])
    acts = legal_actions(ps)
    assert Act.CHECK in acts
    assert Act.BET in acts

def test_engine_check_check_advances_street():
    ps = start_postflop_state(["Ah","Qh"], ["Kh","7d","2h"], to_act=Player.HERO)
    ps = apply_action(ps, Act.CHECK)
    ps = apply_action(ps, Act.CHECK)
    assert ps.gs.street == Street.TURN

def test_engine_bet_fold_terminal():
    ps = start_postflop_state(["Ah","Qh"], ["Kh","7d","2h"], to_act=Player.HERO)
    ps = apply_action(ps, Act.BET)
    ps = apply_action(ps, Act.FOLD)
    assert is_terminal(ps) is True
