from collections import namedtuple, defaultdict
import numba
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32

from AbstractGrid import AbstractGrid
from BinaryGrid import BinaryGrid

Terrain = namedtuple('Terrain', ['id', 'name', 'symbol', 'collidable'])

grass = Terrain(1, 'grass', 'g', None)
rock = Terrain(2, 'rock', 'r', 'rock')
water = Terrain(3, 'water', 'w', 'water')
fence = Terrain(4, 'fence', 'f', 'rock')
wall = Terrain(5, 'wall', 'X', 'wall')


class TerrainGrid(AbstractGrid):
    '''
    This grid is a grid of terrain!
    There can only be one unit of terrain on each tile.

    Terrain can be collidable, but it can only be collidable on one layer at a time.
    '''

    def __init__(self, width, height, default_terrain):
        '''
        collisionTypes: ['rock', 'water']
        '''
        self.width = width
        self.height = height
        self.grid = cuda.device_array([width, height], dtype=int)

        # Terrain Grid
        self.terrains = set()
        self.collidable_terrains = defaultdict(set)

        self.setAllTerrain(default_terrain)

        # Collision Details
        self.collidable_grids = {}

        self.collidable_layers_to_regenerate = set()

        if(default_terrain.collidable):
            self.collidable_layers_to_regenerate.add(default_terrain.collidable)

    def addTerrainType(self, terrain):
        '''
        name: "grass"
        symbol: ".",
        collidiable: None,

        name: "rock"
        symbol: "X",
        collidable: 'rock'
        '''
        self.terrains.add(terrain)
        if terrain.collidable:
            self.collidable_terrains[terrain.collidable].add(terrain.id)

    def setAllTerrain(self, terrain):
        self.addTerrainType(terrain)
        self.setAll(terrain.id)

    def setTerrain(self, x, y, terrain):
        self.addTerrainType(terrain)
        self.grid[x][y] = terrain.id
        if terrain.collidable:
            self.collidable_layers_to_regenerate.add(terrain.collidable)

    def generateCollidables(self):
        for collidable_type in self.collidable_terrains:
            self.generateCollidable(collidable_type)

    def generateCollidable(self, collidable_type):

        @cuda.jit
        def _generateCollidable(grid, newGrid, collidableIdSet):
            x, y = cuda.grid(2)
            if x < array.shape[0] and y < array.shape[1]:
                newGrid[x][y] = grid[x][y] in collidableIdSet

        self.collidable_grids[collidable_type] = BinaryGrid(width, height)
        self.cuda(_generateCollidable)(self.collidable_grids[collidable_type].grid, self.collidable_terrains[collidable_type])

    def tick(self):
        self.generateCollidables()
