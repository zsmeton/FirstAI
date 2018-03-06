# # Imports # #
import numpy as np
import pygame
from matplotlib import pyplot as plt


class Image(pygame.sprite.Sprite):
    def __init__(self, image_file, location):
        pygame.sprite.Sprite.__init__(self)  # call Sprite initializer
        self.image = pygame.image.load(image_file)
        self.rect = self.image.get_rect()
        self.rect.center = location


class Graph:
    def __init__(self, x, array):
        self.x_values = np.linspace(0, x, 1)
        self.y_values = np.array(array)

    def draw(self):
        fig = plt.figure()
        plt.xlabel('Generations')
        plt.ylabel('Fitness')
        plt.plot(self.x_values, self.y_values)
        plt.show()
        print(fig.canvas.get_supported_filetypes())
        fig.savefig('sales.png', transparent=False, dpi=80, bbox_inches="tight")
