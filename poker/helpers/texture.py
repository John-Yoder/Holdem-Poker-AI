from __future__ import annotations
from typing import Dict, Iterable, List, Union

from .cards import Card, parse_cards
from .evaluator import straight_high


def _has_4_to_straight(vals: List[int]) -> bool:
    uniq = sorted(set(vals))
    if 14 in uniq:
        uniq = [1] + uniq
    for start in uniq:
        window = [x for x in uniq if start <= x <= start + 4]
        if len(window) >= 4:
            return True
    return False


def board_texture(board: Iterable[Union[str, Card]]) -> Dict[str, bool]:
    b = parse_cards(board)
    vals = sorted([c.val for c in b], reverse=True)
    suits = [c.suit for c in b]

    # rank texture
    counts = {}
    for v in vals:
        counts[v] = counts.get(v, 0) + 1
    paired = any(n >= 2 for n in counts.values())
    trips_on_board = any(n >= 3 for n in counts.values())

    # suit texture
    suit_counts = {s: suits.count(s) for s in "shdc"}
    max_suit = max(suit_counts.values()) if suit_counts else 0

    # “available” means already present on the board itself
    flush_available = (max_suit >= 5)  # only really possible on river
    flush_draw_available = (max_suit >= 4) and (len(b) < 5)

    # straight texture
    straight_available = straight_high(vals) is not None
    straight_draw_available = (not straight_available) and _has_4_to_straight(vals) and (len(b) < 5)

    # high card descriptors
    high = max(vals)
    ace_high_board = high == 14
    king_high_board = high == 13
    queen_high_board = high == 12
    ten_high_or_lower_board = high <= 10

    # monotone / two-tone (useful for flop logic)
    monotone_flop = (len(b) == 3) and (max_suit == 3)
    two_tone_flop = (len(b) == 3) and (max_suit == 2)

    return {
        "paired_board": paired,
        "trips_on_board": trips_on_board,

        "flush_available": flush_available,
        "flush_draw_available": flush_draw_available,
        "straight_available": straight_available,
        "straight_draw_available": straight_draw_available,

        "ace_high_board": ace_high_board,
        "king_high_board": king_high_board,
        "queen_high_board": queen_high_board,
        "ten_high_or_lower_board": ten_high_or_lower_board,

        "monotone_flop": monotone_flop,
        "two_tone_flop": two_tone_flop,
    }
