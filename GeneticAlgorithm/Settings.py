# # Settings for the program can be set and called from here # #
from GeneticAlgorithm import DNA


def __init__():
    # # Constants # #
    global BLACK, WHITE, GREEN, RED, width, height
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GREEN = (0, 255, 0)
    RED = (255, 0, 0)
    # The desired width and height
    desired_width = 1000
    desired_height = 500
    # actual width and height is calculated so that vector field covers entire screen
    multiple_w = int(desired_width / DNA.amountOfVector)
    multiple_h = int(desired_height / DNA.amountOfVector)
    width = multiple_w * DNA.amountOfVector
    height = multiple_h * DNA.amountOfVector
    print(width, height)
