# The DNA or instructions for each vehicle

# # Import Libraries # #
import math
import random
import numpy as np
import pygame
from pygame import gfxdraw as gfx
from GeneticAlgorithm import Vector, Settings

# # Setup Pygame # #
pygame.init()

# # CONSTANTS # #
TWO_PI = 2 * math.pi

# how large the range of effect is of each vector
amountOfVector = 16

random.seed()
np.random.seed()


class DNA:
    def __init__(self, random_=True):
        self.matrix = np.zeros((math.floor(Settings.width / amountOfVector), math.floor(Settings.height / amountOfVector)))
        if random_:
            for x in np.nditer(self.matrix, op_flags=['readwrite']):
                x[...] = random.uniform(0, TWO_PI)
        dimensions = self.matrix.shape
        self.dimension_x = dimensions[0]
        self.dimension_x -= 1
        self.dimension_y = dimensions[1]
        self.dimension_y -= 1

    # displays the vectors and their range of effect
    def debug(self, screen):
        for index, value in np.ndenumerate(self.matrix):
            starting_point = Vector.Vector(index[0], index[1])
            starting_point.mult(amountOfVector)
            # offsets center so the range of effect is at the corner
            starting_point.add(x_=amountOfVector / 2, y_=amountOfVector / 2)
            starting_point.int()
            # creates vector using the matrix of the dna
            end_point = Vector.Vector.from_angle(angle=value)
            end_point.normalize()
            end_point.mult(7)
            end_point.add(other=starting_point)
            end_point.int()
            # draws the debug
            gfx.aacircle(screen, starting_point.x, starting_point.y, 2, (105, 105, 105))
            pygame.draw.aaline(screen, (105, 105, 105), (starting_point.x, starting_point.y), (end_point.x, end_point.y))

    # returns the vector effecting the object from the position
    def get_vector(self, position):
        x_ = position.x
        y_ = position.y
        x_ -= amountOfVector / 2
        y_ -= amountOfVector / 2
        x_ /= amountOfVector
        y_ /= amountOfVector
        x_ = round(x_)
        y_ = round(y_)
        while x_ > self.dimension_x:
            x_ -= 1
        while y_ > self.dimension_y:
            y_ -= 1
        temp = Vector.Vector.from_angle(self.matrix[int(x_), int(y_)])
        return temp

    # crosses DNA of two individuals
    def cross_over(self, other, mutation_rate=0):
        new_dna = DNA()
        kid = np.nditer(new_dna.matrix, flags=['f_index'], op_flags=['writeonly'])
        mom = np.nditer(self.matrix, flags=['f_index'])
        dad = np.nditer(other.matrix, flags=['f_index'])
        while not mom.finished:
            if Settings.random_list[mom.index] < mutation_rate * 10:
                kid[0] = random.uniform(0, TWO_PI)
            else:
                # 2 options which switch every 6
                parent = math.floor(mom.index/6) % 2
                if parent is 0:
                    kid[0] = mom[0]
                elif parent is 1:
                    kid[0] = dad[0]
                else:
                    print("error")
            kid.iternext()
            mom.iternext()
            dad.iternext()
        return new_dna
