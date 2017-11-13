import math
import timeit
import numba
from numba import cuda




@cuda.jit
def increment_by_one(array):
    x, y = cuda.grid(2)
    if x < array.shape[0] and y < array.shape[1]:
        array[x][y] += 1


# instantiate
start = timeit.default_timer()
cuda_array = numba.cuda.pinned_array([10000, 10000], dtype=int)
stop = timeit.default_timer()
print("Instantiation: ", stop-start)

# increment
start = timeit.default_timer()
threads_per_block = (16, 16)
blocks_per_grid_x = math.ceil(cuda_array.shape[0] / threads_per_block[0])
blocks_per_grid_y = math.ceil(cuda_array.shape[1] / threads_per_block[1])
blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
increment_by_one[blocks_per_grid, threads_per_block](cuda_array)
stop = timeit.default_timer()
print("Increment: ", stop-start)

# print(giant_array[99999])

# access
start = timeit.default_timer()
print(cuda_array[100][100])
stop = timeit.default_timer()

print("Access: ", stop-start)
