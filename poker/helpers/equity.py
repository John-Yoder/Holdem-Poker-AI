from __future__ import annotations
import random
from typing import Iterable, List, Optional, Tuple, Union

from .cards import Card, make_deck, parse_cards
from .evaluator import compare_hands
from .ranges import generate_villain_combos


def monte_carlo_equity(
    hero_hand: Iterable[Union[str, Card]],
    board: Iterable[Union[str, Card]],
    iters: int = 5000,
    villain_hand: Optional[Iterable[Union[str, Card]]] = None,
    villain_range: Optional[Union[str, List[str]]] = None,
    villain_explicit_hands: Optional[List[Iterable[Union[str, Card]]]] = None,
    dead_cards: Optional[Iterable[Union[str, Card]]] = None,
    seed: Optional[int] = None,
) -> Tuple[float, float, float]:
    rng = random.Random(seed)
    hero = parse_cards(hero_hand)
    bd = parse_cards(board)
    dead = parse_cards(dead_cards) if dead_cards else []

    if len(hero) != 2:
        raise ValueError("Hero hand must be 2 cards")
    if not (3 <= len(bd) <= 5):
        raise ValueError("Board must be 3..5 cards post-flop")

    known = hero + bd + dead
    if len(set(known)) != len(known):
        raise ValueError("Duplicate cards in known cards")

    if villain_hand is not None:
        vh = parse_cards(villain_hand)
        if len(vh) != 2:
            raise ValueError("villain_hand must be 2 cards")
        if any(c in set(known) for c in vh) or vh[0] == vh[1]:
            raise ValueError("villain_hand conflicts with known cards")
        villain_combos = [(vh[0], vh[1])]
    else:
        villain_combos = generate_villain_combos(
            exclude=known,
            range_spec=villain_range,
            explicit_hands=villain_explicit_hands,
        )
        if not villain_combos:
            raise ValueError("No valid villain combos from the given range/constraints.")

    wins = ties = losses = 0

    for _ in range(iters):
        vc1, vc2 = villain_combos[rng.randrange(len(villain_combos))]
        iter_known = set(known) | {vc1, vc2}

        deck = make_deck(exclude=iter_known)
        runout = list(bd)
        while len(runout) < 5:
            runout.append(deck.pop(rng.randrange(len(deck))))

        res = compare_hands(hero, [vc1, vc2], runout)
        if res > 0:
            wins += 1
        elif res == 0:
            ties += 1
        else:
            losses += 1

    total = wins + ties + losses
    return wins / total, ties / total, losses / total
