# # Settings for the program can be set and called from here # #
import numpy as np
from GeneticAlgorithm import DNA


BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
width = 0
height = 0
random_list = None
time = 0

def __init__():
    global width, height, random_list
    # # Constants # #
    # CHANGE ONLY THESE TWO WIDTH/HEIGHT VALUES
    desired_width = 1000
    desired_height = 500
    # actual width and height is calculated so that vector field covers entire screen
    multiple_w = int(desired_width / DNA.amountOfVector)
    multiple_h = int(desired_height / DNA.amountOfVector)
    width = multiple_w * DNA.amountOfVector
    height = multiple_h * DNA.amountOfVector
    random_list = 10 * np.random.random((100000,))
    print(width, height)


def new_rand():
    global random_list
    random_list = 10 * np.random.random((100000,))


def timer(reset=False):
    global time
    if reset:
        time = 0
    else:
        time += 1
