import timeit
import time
from numba import cuda

from grid import BinaryGrid, kernel

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


class ConwayGrid(BinaryGrid):

    def step(self):
        start = timeit.default_timer()
        empty_grid = cuda.device_array([self.width, self.height], dtype=bool)
        kernel(conway, self.grid)(self.grid, empty_grid)
        self.grid = empty_grid
        stop = timeit.default_timer()
        print("Step: ", (stop-start)*1000)



start = timeit.default_timer()
bg = ConwayGrid(1000, 1000)
stop = timeit.default_timer()
print("Instantiate: ", stop-start)

print(bg.get(5, 5))

start = timeit.default_timer()
bg.random(seed=time.time())
stop = timeit.default_timer()
print("Randomize: ", stop-start)

for i in range(0, 100):
    bg.step()
bg.samplePrint()
