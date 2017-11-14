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


@cuda.jit
def setRandom(array, xoro_random_states):
    x, y = cuda.grid(2)
    if x < array.shape[0] and y < array.shape[1]:
        array[x][y] = (xoroshiro128p_uniform_float32(xoro_random_states, x) * y % 1) > 0.5


@cuda.jit
def conway(array, newArray):
    x, y = cuda.grid(2)
    if x < array.shape[0] and y < array.shape[1]:

        neighbors = 0
        if (x+1) < array.shape[0] and array[x+1][y]:
            neighbors += 1
        if (y+1) < array.shape[1] and array[x][y+1]:
            neighbors += 1
        if (x-1) >= 0 and array[x-1][y]:
            neighbors += 1
        if (y-1) >= 0 and array[x][y-1]:
            neighbors += 1

        def dead():
            newArray[x][y] = False
        def alive():
            newArray[x][y] = True

        live = array[x][y]
        if live:
            # Any live cell with fewer than two live neighbors dies, as if caused by underpopulation
            if neighbors < 2:
                dead()
            # Any live cell with two or three live neighbors lives on to the next generation
            if neighbors == 2 or neighbors == 3:
                alive()
            # Any live cell with more than three live neighbors dies, as if by overpopulation
            if neighbors > 3:
                dead()
        else:
            # Any dead cell with exactly three live neighbors becomes a live cell, as if by reproduction
            if neighbors == 3:
                alive()
            # Any other dead cell stays dead, Jim
            else:
                dead()


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

    def samplePrint(self):
        for i in range(0, 40):
            for j in range(0, 40):
                if self.get(i, j):
                    print(".", end="")
                else: print("X", end="")
            print("")
        print("-"*40)


class ConwayGrid(BinaryGrid):

    def step(self):
        start = timeit.default_timer()
        empty_grid = numba.cuda.device_array([self.width, self.height], dtype=bool)
        kernel(conway, self.grid)(self.grid, empty_grid)
        self.grid = empty_grid
        stop = timeit.default_timer()
        print("Step: ", (stop-start)*1000)


bg = ConwayGrid(1000, 1000)

print(bg.get(5, 5))

start = timeit.default_timer()
bg.random(seed=time.time())
stop = timeit.default_timer()
print("Randomize: ", stop-start)

for i in range(0, 100):
    bg.step()
bg.samplePrint()
