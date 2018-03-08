# # Imports # #
from GeneticAlgorithm import DNA, Vector, Graphics, Settings, Target
from GeneticAlgorithm.Vector import Vector
import math
import pygame


# Main object which is being optimized
# Creates it as a pygame sprite
class Rocket():
    def __init__(self, DNA_=None):
        self.position = Vector(10, Settings.height/2)
        self.velocity = Vector()
        self.acceleration = Vector()
        self.DNA = DNA.DNA(random_=True)
        if DNA_ is not None:
            self.DNA = DNA_
        self.picture = Graphics.Image('rocket.png', [self.position.x, self.position.y])
        self.fitness = 0
        self.hit_target = False

    def update(self):
        self.acceleration = self.DNA.get_vector(position=self.position)
        if self.acceleration:
            self.acceleration.mult(0.01)
            self.velocity.add(other=self.acceleration)
            self.position.add(other=self.velocity)
            self.picture.rect.center = [self.position.x, self.position.y]
        return self._collision()

    def _collision(self):
        distance = Target.is_reached(self)
        if self.position.x >= Settings.width or self.position.x <= 0:
            return False, False
        elif self.position.y >= Settings.height or self.position.y <= 0:
            return False, False
        elif distance is 1:
            self.hit_target = True
            return False, True
        else:
            return True, False

    def update_fitness(self):
        temp = Target.is_reached(self)
        if self.hit_target:
            a = 10
        else:
            a = 0
        self.fitness = 10 - 10 / (1 + math.exp(- 0.1 * (temp - 30))) + a

    def draw(self, screen):
        screen.blit(self.picture.image, self.picture.rect)


