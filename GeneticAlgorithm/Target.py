# Object file for Target namespace
import pygame
from GeneticAlgorithm import Vector, Settings

position = None
rect = None


def __init__():
    global position, rect
    position = Vector.Vector(Settings.width - 20, Settings.height / 2)
    rect = pygame.Rect(position.x, position.y, 10, 10)
    rect.center = [position.x, position.y]


def draw(screen):
    pygame.draw.rect(screen, (105, 105, 105), rect)


def is_reached(other):
    distance = position.dist(other.position)
    if rect.colliderect(other.picture.rect):
        return 1
    else:
        return distance
