from __future__ import annotations
from itertools import combinations
from typing import Iterable, List, Optional, Tuple, Union

from .cards import Card, RANKS, VAL_TO_RANK, make_deck, parse_cards


def parse_range(range_spec: Union[str, List[str]]) -> List[str]:
    if isinstance(range_spec, list):
        tokens = []
        for x in range_spec:
            tokens.extend([t.strip() for t in x.split(",") if t.strip()])
    else:
        tokens = [t.strip() for t in range_spec.split(",") if t.strip()]

    out: List[str] = []
    for tok in tokens:
        tok = tok.upper()

        if len(tok) == 3 and tok[0] == tok[1] and tok[2] == "+":
            start = RANKS.index(tok[0])
            for r in RANKS[start:]:
                out.append(r + r)
            continue

        if len(tok) == 5 and tok[0] == tok[1] and tok[2] == "-" and tok[3] == tok[4]:
            a, b = tok[0], tok[3]
            ia, ib = RANKS.index(a), RANKS.index(b)
            if ia > ib:
                ia, ib = ib, ia
            for r in RANKS[ia : ib + 1]:
                out.append(r + r)
            continue

        out.append(tok)
    return out


def _is_suited(c1: Card, c2: Card) -> bool:
    return c1.suit == c2.suit


def _combo_matches_pattern(c1: Card, c2: Card, pat: str) -> bool:
    v1, v2 = c1.val, c2.val
    r1, r2 = VAL_TO_RANK[v1], VAL_TO_RANK[v2]
    hi, lo = (r1, r2) if v1 >= v2 else (r2, r1)

    if len(pat) == 2 and pat[0] == pat[1]:
        return hi == pat[0] and lo == pat[1]

    if len(pat) == 3 and pat[2] in ("s", "o"):
        suited = pat[2] == "s"
        if {hi, lo} != {pat[0], pat[1]}:
            return False
        return _is_suited(c1, c2) if suited else (not _is_suited(c1, c2))

    if len(pat) == 2:
        return {hi, lo} == {pat[0], pat[1]}

    return False


def all_remaining_two_card_combos(exclude: Iterable[Card]) -> List[Tuple[Card, Card]]:
    deck = make_deck(exclude=exclude)
    return list(combinations(deck, 2))


def generate_villain_combos(
    exclude: Iterable[Card],
    range_spec: Optional[Union[str, List[str]]] = None,
    explicit_hands: Optional[List[Iterable[Union[str, Card]]]] = None,
) -> List[Tuple[Card, Card]]:
    dead = set(exclude)

    if explicit_hands is not None:
        combos = []
        for h in explicit_hands:
            cs = parse_cards(h)
            if len(cs) != 2:
                raise ValueError("Explicit villain hand must be 2 cards")
            if cs[0] in dead or cs[1] in dead or cs[0] == cs[1]:
                continue
            combos.append((cs[0], cs[1]))
        return combos

    all_combos = all_remaining_two_card_combos(dead)
    if range_spec is None:
        return all_combos

    pats = parse_range(range_spec)
    out = []
    for c1, c2 in all_combos:
        for pat in pats:
            if _combo_matches_pattern(c1, c2, pat):
                out.append((c1, c2))
                break
    return out
