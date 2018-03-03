# Main Looping File
# Sets up population
# Draws the current generation

# # Import Libraries # #
import Vector  # vector libray similar to PVector
import pygame  # drawing and event handling

# # Constants # #
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# # Pygame Setup # #
pygame.init()
width = 500
height = 200
screen = pygame.display.set_mode([width, height])
pygame.display.set_caption("Genetic Path Finding")  # name of the window created
clock = pygame.time.Clock()  # used to manage how fast the screen updates


