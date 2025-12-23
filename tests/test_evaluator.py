from poker.helpers.evaluator import compare_hands, evaluate_best

def test_straight_flush_beats_flush():
    board = ["Qs","Js","Ts","2d","3c"]
    hero = ["As","Ks"]      # royal
    vill = ["Ah","Kh"]      # just broadway straight (not flush)
    assert compare_hands(hero, vill, board) == 1

def test_pair_vs_high_card():
    board = ["2s","7d","Jh","4c","9c"]
    hero = ["Jc","3d"]      # pair of J
    vill = ["Ah","Kd"]      # high card
    assert compare_hands(hero, vill, board) == 1

def test_evaluate_best_returns_expected_shape():
    name, tb, best5 = evaluate_best(["Ah","Ad"], ["7c","8d","9s"])
    assert isinstance(name, str)
    assert isinstance(tb, tuple)
    assert len(best5) == 5
