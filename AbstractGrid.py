from collections import namedtuple, defaultdict

import math
import timeit
import functools

import numba
from numba import cuda


def gridKernel(operation, array):
    '''
    Putting together a CUDA kernel for a grid can be kinda tough
    so here are some sensible defaults for a grid
    '''
    threads_per_block = (16, 16)
    blocks_per_grid_x = math.ceil(array.shape[0] / threads_per_block[0])
    blocks_per_grid_y = math.ceil(array.shape[1] / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    return operation[blocks_per_grid, threads_per_block]


class AbstractGrid:

    def get(self, x, y):
        return self.grid[x][y]

    def cuda(self, fn):
        '''
        To call a CUDA function on this grid, use

        > self.cuda(fn)(args)

        for example

        > self.cuda(setValue)(True)
        '''
        return functools.partial(gridKernel(fn, self.grid), self.grid)


    def setAll(self, value):
        '''
        Sets every coordinate in the entire grid to value.
        '''
        @cuda.jit
        def setValue(array, val):
            x, y = cuda.grid(2)
            if x < array.shape[0] and y < array.shape[1]:
                array[x][y] = val

        self.cuda(setValue)(value)

    def setAllWithMask(self, value, mask):
        '''
        mask is a BinaryGrid that has the same dimensions as this grid (or larger)
        For every tile that is truthy in the mask, set value on this grid
        '''
        @cuda.jit
        def setValue(array, val, mask):
            x, y = cuda.grid(2)
            if x < array.shape[0] and y < array.shape[1] and mask[x][y]:
                array[x][y] = val

        self.cuda(setValue)(value, mask)
