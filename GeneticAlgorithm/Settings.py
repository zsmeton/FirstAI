# # Settings for the program can be set and called from here # #
from GeneticAlgorithm import DNA

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
width = 0
height = 0


def __init__():
    global width, height
    # # Constants # #
    # CHANGE ONLY THESE TWO WIDTH/HEIGHT VALUES
    desired_width = 1000
    desired_height = 500
    # actual width and height is calculated so that vector field covers entire screen
    multiple_w = int(desired_width / DNA.amountOfVector)
    multiple_h = int(desired_height / DNA.amountOfVector)
    width = multiple_w * DNA.amountOfVector
    height = multiple_h * DNA.amountOfVector
    print(width, height)
