import math
import timeit
import numba
import time
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32

from functools import partial


def kernel(operation, array):
    threads_per_block = (16, 16)
    blocks_per_grid_x = math.ceil(array.shape[0] / threads_per_block[0])
    blocks_per_grid_y = math.ceil(array.shape[1] / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    return operation[blocks_per_grid, threads_per_block]


@cuda.jit
def set(array, val):
    x, y = cuda.grid(2)
    if x < array.shape[0] and y < array.shape[1]:
        array[x][y] = val


@cuda.jit(device=True)
def coinToss(xoro_random_states, x, y):
    return xoroshiro128p_uniform_float32(xoro_random_states, x*y) > 0.5

@cuda.jit
def setRandom(array, xoro_random_states):
    x, y = cuda.grid(2)
    if x < array.shape[0] and y < array.shape[1]:
        array[x][y] = (xoroshiro128p_uniform_float32(xoro_random_states, x) * y % 1) > 0.5


class Grid:
    def get(self, x, y):
        return self.grid[x][y]

    def slice(self, x1, y1, x2, y2):
        # TODO: this
        pass


class BinaryGrid(Grid):
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = numba.cuda.device_array([width, height], dtype=bool)

    def clear(self):
        kernel(set, self.grid)(self.grid, False)

    def mask(self):
        kernel(set, self.grid)(self.grid, True)

    def random(self, seed=1):
        rng_states = create_xoroshiro128p_states(self.width, seed=seed)
        kernel(setRandom, self.grid)(self.grid, rng_states)



bg = BinaryGrid(1000, 1000)

print(bg.get(5, 5))

start = timeit.default_timer()
bg.random(seed=time.time())
stop = timeit.default_timer()
print("Randomize: ", stop-start)

for i in range(0, 40):
    for j in range(0, 40):
        if bg.get(i, j):
            print(".", end="")
        else: print("X", end="")
    print("")
