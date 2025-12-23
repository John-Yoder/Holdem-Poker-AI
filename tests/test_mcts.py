import random

from poker.engine.simple_hu_postflop import start_postflop_state, legal_actions
from poker.ai.mcts import MCTS
from poker.helpers.abstraction import Act

def test_mcts_returns_legal_action():
    ps = start_postflop_state(["Ah","Qh"], ["Kh","7d","2h"])
    mcts = MCTS(rng=random.Random(1), rollout_cache_capacity=10_000, rollout_cache_min_samples=3)
    act, stats = mcts.search(ps, iters=200)
    assert act in legal_actions(ps)
    assert isinstance(stats, dict)
    for k, v in stats.items():
        assert k in legal_actions(ps)
        assert -1.0 <= v <= 1.0

def test_mcts_stats_keys_are_actions():
    ps = start_postflop_state(["Ah","Qh"], ["Kh","7d","2h"])
    mcts = MCTS(rng=random.Random(2))
    act, stats = mcts.search(ps, iters=100)
    assert all(isinstance(a, int) for a in stats.keys())
