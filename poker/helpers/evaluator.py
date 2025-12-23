from __future__ import annotations
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Tuple, Union

from .cards import Card, parse_cards

CATEGORY = {
    "high_card": 0,
    "pair": 1,
    "two_pair": 2,
    "trips": 3,
    "straight": 4,
    "flush": 5,
    "full_house": 6,
    "quads": 7,
    "straight_flush": 8,
}


def _rank_counts(cards: List[Card]) -> Dict[int, int]:
    d: Dict[int, int] = {}
    for c in cards:
        d[c.val] = d.get(c.val, 0) + 1
    return d


def straight_high(values: List[int]) -> Optional[int]:
    uniq = sorted(set(values), reverse=True)
    if 14 in uniq:
        uniq.append(1)  # ace low
    run = 1
    best = None
    for i in range(len(uniq) - 1):
        if uniq[i] - 1 == uniq[i + 1]:
            run += 1
            if run >= 5:
                high = uniq[i - (run - 2)]
                best = max(best or 0, high)
        else:
            run = 1
    if best == 1:
        return 5
    return best


def evaluate_5(cards5: List[Card]) -> Tuple[int, Tuple[int, ...], str]:
    if len(cards5) != 5:
        raise ValueError("evaluate_5 expects exactly 5 cards")

    vals = sorted([c.val for c in cards5], reverse=True)
    suits = [c.suit for c in cards5]
    is_flush = len(set(suits)) == 1

    counts = _rank_counts(cards5)
    groups = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    count_pattern = sorted(counts.values(), reverse=True)

    sh = straight_high(vals)
    is_straight = sh is not None

    if is_straight and is_flush:
        return CATEGORY["straight_flush"], (sh,), "straight_flush"
    if count_pattern == [4, 1]:
        quad = groups[0][0]
        kicker = max(v for v in vals if v != quad)
        return CATEGORY["quads"], (quad, kicker), "quads"
    if count_pattern == [3, 2]:
        trips = groups[0][0]
        pair = groups[1][0]
        return CATEGORY["full_house"], (trips, pair), "full_house"
    if is_flush:
        return CATEGORY["flush"], tuple(vals), "flush"
    if is_straight:
        return CATEGORY["straight"], (sh,), "straight"
    if count_pattern == [3, 1, 1]:
        trips = groups[0][0]
        kickers = sorted([v for v in vals if v != trips], reverse=True)
        return CATEGORY["trips"], (trips, *kickers), "trips"
    if count_pattern == [2, 2, 1]:
        pair_hi = groups[0][0]
        pair_lo = groups[1][0]
        kicker = max(v for v in vals if v != pair_hi and v != pair_lo)
        return CATEGORY["two_pair"], (pair_hi, pair_lo, kicker), "two_pair"
    if count_pattern == [2, 1, 1, 1]:
        pair = groups[0][0]
        kickers = sorted([v for v in vals if v != pair], reverse=True)
        return CATEGORY["pair"], (pair, *kickers), "pair"
    return CATEGORY["high_card"], tuple(vals), "high_card"


def evaluate_best(
    hand: Iterable[Union[str, Card]],
    board: Iterable[Union[str, Card]],
) -> Tuple[str, Tuple[int, ...], List[Card]]:
    h = parse_cards(hand)
    b = parse_cards(board)
    cards = h + b
    if len(h) != 2:
        raise ValueError("Hold'em hand must be exactly 2 cards")
    if not (3 <= len(b) <= 5):
        raise ValueError("Board must be 3, 4, or 5 cards post-flop")
    if len(set(cards)) != len(cards):
        raise ValueError("Duplicate cards detected")

    best = None  # (cat_rank, tiebreak, name, best5)
    for combo in combinations(cards, 5):
        cat_rank, tiebreak, name = evaluate_5(list(combo))
        key = (cat_rank, tiebreak)
        if best is None or key > (best[0], best[1]):
            best = (cat_rank, tiebreak, name, list(combo))

    assert best is not None
    best5 = sorted(best[3], key=lambda c: c.val, reverse=True)
    return best[2], best[1], best5


def compare_hands(hand1, hand2, board) -> int:
    n1, t1, _ = evaluate_best(hand1, board)
    n2, t2, _ = evaluate_best(hand2, board)
    k1 = (CATEGORY[n1], t1)
    k2 = (CATEGORY[n2], t2)
    return 1 if k1 > k2 else (-1 if k2 > k1 else 0)


def winners(hands, board) -> List[int]:
    keys = []
    for i, h in enumerate(hands):
        name, tiebreak, _ = evaluate_best(h, board)
        keys.append((CATEGORY[name], tiebreak, i))
    best = max(keys, key=lambda x: (x[0], x[1]))
    bc, bt = best[0], best[1]
    return [i for (c, tb, i) in keys if (c, tb) == (bc, bt)]
