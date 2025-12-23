from __future__ import annotations
from typing import Dict, Iterable, Tuple, Union

from .cards import Card, parse_cards
from .evaluator import CATEGORY, evaluate_best
from .texture import board_texture
from .draws import draw_features  # (we’ll define in a second)


def made_hand_features(hand, board) -> Dict[str, bool]:
    h = parse_cards(hand)
    b = parse_cards(board)
    bv = sorted([c.val for c in b], reverse=True)

    top = max(bv)
    mid = bv[1] if len(bv) >= 2 else None
    bot = bv[-1]

    hv = sorted([c.val for c in h], reverse=True)
    hi, lo = hv[0], hv[1]

    pocket_pair = hi == lo
    overpair = pocket_pair and hi > top
    underpair = pocket_pair and hi < bot

    pair_with_board = (hi in bv) or (lo in bv)
    top_pair = (hi == top) or (lo == top)
    middle_pair = (mid is not None) and ((hi == mid) or (lo == mid)) and not top_pair
    bottom_pair = ((hi == bot) or (lo == bot)) and not top_pair and not middle_pair

    two_overcards = (hi > top and lo > top)
    one_overcard = (hi > top and lo <= top) or (lo > top and hi <= top)

    name, _, _ = evaluate_best(h, b)

    return {
        "pocket_pair": pocket_pair,
        "overpair": overpair,
        "underpair": underpair,
        "pair_with_board": pair_with_board,
        "top_pair": top_pair,
        "middle_pair": middle_pair,
        "bottom_pair": bottom_pair,
        "two_overcards": two_overcards,
        "one_overcard": one_overcard,
        "made_pair_or_better": CATEGORY[name] >= CATEGORY["pair"],
        "made_two_pair_or_better": CATEGORY[name] >= CATEGORY["two_pair"],
        "made_trips_or_better": CATEGORY[name] >= CATEGORY["trips"],
    }


def extract_features(hand, board) -> Dict[str, Union[bool, int, float, Tuple[int, ...], str]]:
    h = parse_cards(hand)
    b = parse_cards(board)
    name, tiebreak, _ = evaluate_best(h, b)

    feats: Dict[str, Union[bool, int, float, Tuple[int, ...], str]] = {}
    feats.update(board_texture(b))
    feats.update(made_hand_features(h, b))
    feats.update(draw_features(h, b))

    feats["made_hand_category"] = name
    feats["made_hand_rank"] = CATEGORY[name]
    feats["tiebreak"] = tiebreak
    return feats
