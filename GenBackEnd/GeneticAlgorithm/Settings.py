# # Settings for the program can be set and called from here # #
from GenBackEnd.GeneticAlgorithm import DNA

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
width = 0
height = 0
random_list = None
time = 0
max_time = 2000
min_time = max_time


def __init__():
    global width, height, random_list, min_time
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
    with open('stats.txt', 'r+') as file:
        contents = file.read()
        if contents.isdecimal():
            min_time = float(contents)
        else:
            print("ERROR: CANNOT READ STATS.TXT")


def new_min(value):
    global min_time
    min_time = value
    with open('stats.txt', 'w+') as file:
        file.write(str(min_time))


def timer(reset=False):
    global time
    if reset:
        time = 0
    else:
        time += 1

