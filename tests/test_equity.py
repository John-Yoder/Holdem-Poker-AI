from poker.helpers.equity import monte_carlo_equity

def test_equity_reasonable_bounds():
    wr, tr, lr = monte_carlo_equity(["Ah","Ad"], ["7c","8d","9s"], iters=2000, seed=1)
    assert 0.0 <= wr <= 1.0
    assert 0.0 <= tr <= 1.0
    assert 0.0 <= lr <= 1.0
    assert abs((wr+tr+lr) - 1.0) < 1e-6

def test_equity_vs_specific_hand_runs():
    wr, tr, lr = monte_carlo_equity(
        ["Ah","Qh"], ["Kh","7d","2h"],
        iters=2000, villain_hand=["Kc","Kd"], seed=2
    )
    assert 0.0 <= wr <= 1.0
