from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Union

RANKS = "23456789TJQKA"
SUITS = "shdc"
RANK_TO_VAL = {r: i + 2 for i, r in enumerate(RANKS)}  # 2..14
VAL_TO_RANK = {v: r for r, v in RANK_TO_VAL.items()}


@dataclass(frozen=True, order=True)
class Card:
    val: int
    suit: str

    def __str__(self) -> str:
        return f"{VAL_TO_RANK[self.val]}{self.suit}"

    @staticmethod
    def from_str(s: str) -> "Card":
        s = s.strip()
        if len(s) != 2:
            raise ValueError(f"Bad card string: {s!r}")
        r, su = s[0].upper(), s[1].lower()
        if r not in RANK_TO_VAL or su not in SUITS:
            raise ValueError(f"Bad card string: {s!r}")
        return Card(RANK_TO_VAL[r], su)


def parse_cards(cards: Iterable[Union[str, Card]]) -> List[Card]:
    out: List[Card] = []
    for x in cards:
        out.append(x if isinstance(x, Card) else Card.from_str(x))
    return out


def make_deck(exclude: Iterable[Card] = ()) -> List[Card]:
    dead = set(exclude)
    deck = [Card(RANK_TO_VAL[r], s) for r in RANKS for s in SUITS]
    return [c for c in deck if c not in dead]
