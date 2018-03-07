# # Imports # #
from GeneticAlgorithm import DNA, Vector, Graphics, Settings
import math
import pygame

# Main object which is being optimized
class Rocket:
    def __init__(self, DNA_=None):
        self.position = Vector.Vector(10, 248)
        self.velocity = Vector.Vector()
        self.acceleration = Vector.Vector()
        self.DNA = DNA.DNA(random_=True)
        if DNA_ is not None:
            self.DNA = DNA_
        self.picture = Graphics.Image('rocket.png', [self.position.x, self.position.y])
        self.fitness = 0
        self.target = Vector.Vector(972, 248)
        self.hit_target = False

    def update(self):
        self.acceleration = self.DNA.get_vector(self.position)
        if self.acceleration:
            self.acceleration.mult(0.01)
            self.velocity.add(self.acceleration)
            self.position.add(self.velocity)
            self.picture.rect.center = [self.position.x, self.position.y]
        return self._collision()

    def _collision(self):
        close = self.position.dist(self.target)
        if self.position.x >= Settings.width or self.position.x <= 0:
            return False, False
        elif self.position.y >= Settings.height or self.position.y <= 0:
            return False, False
        elif close < 20:
            self.hit_target = True
            return False, True
        else:
            return True, False

    def update_fitness(self):
        temp = self.position.dist(self.target)
        if self.hit_target:
            a = 10
        else:
            a = 0
        self.fitness = 10 - 10/(1 + math.exp(- 0.1 * (temp - 30))) + a

    def draw(self, screen):
        screen.blit(self.picture.image, self.picture.rect)


# Target for the rockets
class Target:
    def __init__(self):
        self.position = Vector.Vector(Settings.width - 20, Settings.height / 2)
        self.rect = pygame.Rect(self.position.x, self.position.y, 10, 10)
        self.rect.center = [self.position.x, self.position.y]

    def draw(self, screen):
        pygame.draw.rect(screen, (105, 105, 104), self.rect)
