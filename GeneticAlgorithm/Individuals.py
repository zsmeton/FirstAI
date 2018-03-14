# # Imports # #
import pygame
from scipy.special import expit

from GeneticAlgorithm import DNA, Vector, Graphics, Settings, Target, Statist


# Main object which is being optimized
class Rocket:
    def __init__(self, DNA_=None):
        self.position = Vector.Vector(10, Settings.height / 2)
        self.velocity = Vector.Vector()
        self.acceleration = Vector.Vector()
        self.DNA = DNA.DNA(DNA_)
        self.picture = Graphics.Image('rocket.png', [self.position.x, self.position.y])
        self.fitness = 0
        self.hit_target = False
        self.hit_time = Settings.max_time
        self.time_fitness = 0
        self.dist = Settings.width
        self.dist_fitness = 0
        self.rect = self.picture.rect
        self.alive = True

    def update(self, obstacles):
        self.acceleration = self.DNA.get_vector(position=self.position)
        if self.acceleration:
            self.acceleration.mult(0.01)
            self.velocity.add(other=self.acceleration)
            self.position.add(other=self.velocity)
            self.picture.rect.center = [self.position.x, self.position.y]
            self.rect = self.picture.rect
        return self._collision(obstacles)

    def _collision(self, obstacles):
        distance = Target.is_reached(self)
        if self.position.x >= Settings.width or self.position.x <= 0:
            self.alive = False
            return False, False
        elif self.position.y >= Settings.height or self.position.y <= 0:
            self.alive = False
            return False, False
        elif distance is 1:
            self.hit_time = Settings.time
            self.hit_target = True
            return False, True
        elif obstacles.collide(self):
            self.alive = False
            return False, False
        else:
            return True, False

    def update_fitness(self):
        self.dist = Target.is_reached(self)
        if self.hit_target:
            a = 2
            if self.hit_time < Settings.min_time:
                Settings.new_min(self.hit_time)
        else:
            a = 1
        b = 1
        if not self.alive:
            b = 0.8
        # map time to same value as distance
        time = Statist.variable_mapping(self.hit_time, Settings.min_time, Settings.max_time, Target.size,
                                        Settings.width)
        # find the fitness of each using inverse
        # Source : https://gamedev.stackexchange.com/questions/17620/equation-to-make-small-number-big-and-big-number-small-gravity
        self.dist_fitness = (300 / (self.dist + .1))
        self.time_fitness = (300 / (time + .1))
        # use prioritized fitness algorithm
        # Source : https://geekyisawesome.blogspot.com/2013/06/fitness-function-for-multi-objective.html
        # print("Fitness of Dist: %.2f\t Time: %.2f" % (self.dist_fitness, self.time_fitness))
        self.fitness = a * b * (self.dist_fitness * expit(self.time_fitness))

    def draw(self, screen):
        screen.blit(self.picture.image, self.picture.rect)


class Obstacle:
    def __init__(self, topleft, bottomright):
        width_hieght = bottomright - topleft
        width, height = width_hieght.x, width_hieght.y
        self.rect = pygame.Rect(topleft.x, topleft.y, width, height)
        if width < 0:
            width = topleft.x - bottomright.x
            self.rect.width = width
            self.rect.left = topleft.x - width
        if height < 0:
            height = topleft.y - bottomright.y
            self.rect.height = height
            self.rect.top = topleft.y - height

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                return True

    def draw(self, screen, color=(105, 105, 105)):
        pygame.draw.rect(screen, color, self.rect)

    def collide(self, other):
        if self.rect.colliderect(other.rect):
            return True
