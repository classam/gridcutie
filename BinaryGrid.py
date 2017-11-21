from collections import namedtuple, defaultdict

import math
import timeit
import functools

import numba
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32

from AbstractGrid import AbstractGrid

class BinaryGrid(AbstractGrid):
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = cuda.device_array([width, height], dtype=bool)

    def clear(self):
        self.setAll(False)

    def mask(self):
        self.setAll(True)

    def random(self, seed=1):
        @cuda.jit
        def setRandom(array, xoro_random_states):
            x, y = cuda.grid(2)
            if x < array.shape[0] and y < array.shape[1]:
                array[x][y] = (xoroshiro128p_uniform_float32(xoro_random_states, x) * y % 1) > 0.5

        rng_states = create_xoroshiro128p_states(self.width, seed=seed)
        self.cuda(setRandom)(rng_states)

    def samplePrint(self):
        for i in range(0, 40):
            for j in range(0, 40):
                if self.get(i, j):
                    print("X", end="")
                else: print(".", end="")
            print("")
        print("-"*40)
