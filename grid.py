import math
import timeit
import numba
from numba import cuda
from functools import partial


def cudaMap(array, operation):
    threads_per_block = (16, 16)
    blocks_per_grid_x = math.ceil(cuda_array.shape[0] / threads_per_block[0])
    blocks_per_grid_y = math.ceil(cuda_array.shape[1] / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    operation[blocks_per_grid, threads_per_block](array)


def cudaApply(array, otherArray, operation):
    threads_per_block = (16, 16)
    blocks_per_grid_x = math.ceil(cuda_array.shape[0] / threads_per_block[0])
    blocks_per_grid_y = math.ceil(cuda_array.shape[1] / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    operation[blocks_per_grid, threads_per_block](array, otherArray)


@cuda.jit(device=True)
def incr(array, x, y):
    return array[x][y] + 1


@cuda.jit
def increment_by_one(array):
    x, y = cuda.grid(2)
    if x < array.shape[0] and y < array.shape[1]:
        array[x][y] = incr(array, x, y)


@cuda.jit
def add_two_arrays(array, otherArray):
    x, y = cuda.grid(2)
    if x < array.shape[0] and y < array.shape[1]:
        array[x][y] = array[x][y] + otherArray[x][y]


@cuda.jit
def set(array, val):
    x, y = cuda.grid(2)
    if x < array.shape[0] and y < array.shape[1]:
        array[x][y] = val


class BinaryGrid:
    def __init__(self, x, y):
        self.grid = numba.cuda.device_array([x, y], dtype=bool)

    def clear(self):
        cudaMap(self.grid, partial(val=0))


# instantiate
start = timeit.default_timer()
cuda_array = numba.cuda.device_array([10000, 10000], dtype=int)
cuda_array_2 = numba.cuda.device_array([10000, 10000], dtype=int)
stop = timeit.default_timer()
print("Instantiation: ", stop-start)

# increment
start = timeit.default_timer()
cudaMap(cuda_array, increment_by_one)
cudaMap(cuda_array, increment_by_one)
cudaMap(cuda_array_2, increment_by_one)
cudaApply(cuda_array, cuda_array_2, add_two_arrays)
stop = timeit.default_timer()
print("Increment: ", stop-start)

# print(giant_array[99999])

# access
start = timeit.default_timer()
print("Array 1:", cuda_array[100][100])
print("Array 2:", cuda_array_2[100][100])
stop = timeit.default_timer()

print("Access: ", stop-start)
