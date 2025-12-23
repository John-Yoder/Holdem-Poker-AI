from __future__ import annotations
from typing import Dict, Iterable, List, Optional, Tuple, Union

from .cards import Card, make_deck, parse_cards
from .evaluator import straight_high


def straight_outs_next_card(hand_cards: List[Card], board_cards: List[Card]) -> Tuple[int, bool, bool]:
    cards = hand_cards + board_cards
    known = set(cards)
    deck = make_deck(exclude=known)

    current_vals = [c.val for c in cards]
    outs = set()

    for c in deck:
        vals = current_vals + [c.val]
        if straight_high(vals) is not None:
            outs.add(c.val)

    outs_count = len(outs)
    # heuristic classification
    if outs_count >= 2:
        return outs_count, True, False
    if outs_count == 1:
        return outs_count, False, True
    return 0, False, False


def _backdoor_flush_prob_flop(cards: List[Card]) -> float:
    suit_counts = {s: 0 for s in "shdc"}
    for c in cards:
        suit_counts[c.suit] += 1
    if max(suit_counts.values()) >= 4:
        return 0.0

    prob = 0.0
    for s in "shdc":
        have = suit_counts[s]
        need = 5 - have
        if need == 2:
            rem = 13 - have
            if rem >= 2:
                prob += (rem / 47.0) * ((rem - 1) / 46.0)
    return prob


def _backdoor_straight_prob_flop(hand: List[Card], board: List[Card]) -> float:
    cards = hand + board
    known = set(cards)
    deck = make_deck(exclude=known)

    outs_now, _, _ = straight_outs_next_card(hand, board)
    if outs_now > 0:
        return 0.0

    total = 47 * 46
    hit = 0
    base_vals = [c.val for c in cards]

    for i, turn in enumerate(deck):
        remaining = deck[:i] + deck[i + 1 :]
        vals_turn = base_vals + [turn.val]
        for river in remaining:
            vals = vals_turn + [river.val]
            if straight_high(vals) is not None:
                hit += 1
    return hit / total


def draw_features(hand, board) -> Dict[str, Union[bool, int, float]]:
    h = parse_cards(hand)
    b = parse_cards(board)
    cards = h + b
    known = set(cards)
    deck = make_deck(exclude=known)

    # flush draw outs next card
    suit_counts = {s: 0 for s in "shdc"}
    for c in cards:
        suit_counts[c.suit] += 1
    best_suit = max(suit_counts, key=lambda s: suit_counts[s])
    max_count = suit_counts[best_suit]

    has_flush_draw = (max_count == 4) and (len(b) < 5)
    outs_flush_turn = sum(1 for c in deck if c.suit == best_suit) if has_flush_draw else 0

    outs_straight_turn, oesd, gutshot = straight_outs_next_card(h, b)

    backdoor_flush_prob = _backdoor_flush_prob_flop(cards) if len(b) == 3 else 0.0
    backdoor_straight_prob = _backdoor_straight_prob_flop(h, b) if len(b) == 3 else 0.0

    combo_draw = (outs_flush_turn > 0) and (outs_straight_turn > 0)

    return {
        "has_flush_draw": has_flush_draw,
        "outs_flush_turn": outs_flush_turn,
        "has_straight_draw": outs_straight_turn > 0,
        "outs_straight_turn": outs_straight_turn,
        "is_oesd": oesd,
        "is_gutshot": gutshot,
        "combo_draw": combo_draw,
        "backdoor_flush_prob": backdoor_flush_prob,
        "backdoor_straight_prob": backdoor_straight_prob,
    }
