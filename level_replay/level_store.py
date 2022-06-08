"""Code from https://github.com/ucl-dark/paired.
Only for proper loading of the saved model."""
from collections import namedtuple, defaultdict, deque

import numpy as np
import torch

INT32_MAX = 2147483647


class LevelStore(object):
    """
    Manages a mapping between level index --> level, where the level
    may be represented by any arbitrary data structure. Typically, we can
    represent any given level as a string.
    """

    def __init__(self, max_size=None):
        self.max_size = max_size
        self.seed2level = defaultdict()
        self.level2seed = defaultdict()
        self.next_seed = 1
        self.levels = set()

    def __len__(self):
        return len(self.levels)

    def _insert(self, level):
        if level is None:
            return None

        # FIFO if max size constraint
        if self.max_size is not None:
            while len(self.levels) >= self.max_size:
                first_idx = list(self.seed2level)[0]
                self._remove(first_idx)

        if level not in self.levels:
            seed = self.next_seed
            self.seed2level[seed] = level
            self.level2seed[level] = seed
            self.levels.add(level)
            self.next_seed += 1
            return seed
        else:
            return self.level2seed[level]

    def insert(self, level):
        if hasattr(level, "__iter__"):
            idx = []
            for l in level:
                idx.append(self._insert(l))
            return idx
        else:
            return self._insert(level)

    def _remove(self, level_seed):
        if level_seed is None or level_seed < 0:
            return

        level = self.seed2level[level_seed]
        self.levels.remove(level)
        del self.seed2level[level_seed]
        del self.level2seed[level]

    def remove(self, level_seed):
        if hasattr(level_seed, "__iter__"):
            for i in level_seed:
                self._remove(i)
        else:
            self._remove(level_seed)

    def reconcile_seeds(self, level_seeds):
        old_seeds = set(self.seed2level)
        new_seeds = set(level_seeds)

        # Don't update if empty seeds
        if len(new_seeds) == 1 and -1 in new_seeds:
            return

        ejected_seeds = old_seeds - new_seeds
        for seed in ejected_seeds:
            self._remove(seed)

    def get_level(self, level_seed):
        return self.seed2level[level_seed]
