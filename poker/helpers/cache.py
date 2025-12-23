from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
from typing import Dict, Generic, Hashable, Optional, Tuple, TypeVar

K = TypeVar("K", bound=Hashable)


@dataclass(slots=True)
class RolloutStats:
    """
    Tracks running mean EV for a bucket/state key.
    EV convention suggestion:
      +1.0 = win, 0.0 = tie, -1.0 = loss
    """
    n: int = 0
    mean_ev: float = 0.0

    def update(self, ev: float) -> None:
        self.n += 1
        # incremental mean
        self.mean_ev += (ev - self.mean_ev) / self.n


class LRUTranspoTable(Generic[K]):
    """
    LRU cache that stores RolloutStats per key.
    Great for:
      - MCTS rollout value caching
      - abstraction bucket caching
    """
    def __init__(self, capacity: int = 200_000):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._od: "OrderedDict[K, RolloutStats]" = OrderedDict()

        # cheap instrumentation
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def __len__(self) -> int:
        return len(self._od)

    def get(self, key: K) -> Optional[RolloutStats]:
        v = self._od.get(key)
        if v is None:
            self.misses += 1
            return None
        self.hits += 1
        # mark as recently used
        self._od.move_to_end(key, last=True)
        return v

    def get_or_create(self, key: K) -> RolloutStats:
        v = self.get(key)
        if v is not None:
            return v
        # create new entry
        v = RolloutStats()
        self._od[key] = v
        self._od.move_to_end(key, last=True)
        self._evict_if_needed()
        return v

    def update(self, key: K, ev: float) -> RolloutStats:
        v = self.get_or_create(key)
        v.update(ev)
        return v

    def _evict_if_needed(self) -> None:
        while len(self._od) > self.capacity:
            self._od.popitem(last=False)  # least recently used
            self.evictions += 1

    def clear(self) -> None:
        self._od.clear()
        self.hits = self.misses = self.evictions = 0
