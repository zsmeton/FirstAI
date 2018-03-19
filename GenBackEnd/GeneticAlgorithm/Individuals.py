# # Imports # #
import pygame

from GenBackEnd.GeneticAlgorithm import Graphics, Vector, DNA, Settings, Statist, Target


# Main object which is being optimized
class Rocket:
    def __init__(self, DNA_=None):
        self.position = Vector.Vector(10, Settings.height / 2)
        self.velocity = Vector.Vector()
        self.acceleration = Vector.Vector()
        self.DNA = DNA.DNA(DNA_)
        self.picture = Graphics.Image('rocket.png', [self.position.x, self.position.y])
        self.fitness = 0
        self.hit_time = Settings.max_time
        self.time_fitness = 0
        self.dist = Settings.width
        self.dist_fitness = 0
        self.rect = self.picture.rect
        self.alive = True
        self.reached_target = False

    def update(self):
        self.acceleration = self.DNA.get_vector(position=self.position)
        if self.acceleration:
            self.acceleration.mult(0.01)
            self.velocity.add(other=self.acceleration)
            self.position.add(other=self.velocity)
            self.picture.rect.center = [self.position.x, self.position.y]
            self.rect = self.picture.rect

    def collision(self, obstacles):
        if self.hit_wall():
            self.alive = False
        elif Target.is_reached(self) is 1:
            self.hit_time = Settings.time
            self.alive = False
            self.reached_target = True
        elif obstacles.collide(self):
            self.alive = False

    def hit_wall(self):
        if self.position.x >= Settings.width or self.position.x <= 0:
            return True
        elif self.position.y >= Settings.height or self.position.y <= 0:
            return True
        else:
            return False

    def update_fitness(self):
        self.dist = Target.is_reached(self)
        a = 1
        b = 1
        if self.reached_target:
            a = 1.2
            if self.hit_time < Settings.min_time:
                Settings.new_min(self.hit_time)
        elif not self.alive:
            b = 0.9
        # map time to same value as distance
        time = Statist.variable_mapping(self.hit_time, Settings.min_time, Settings.max_time, Target.size, Settings.width)
        # find the fitness of each using inverse
        # Source : https://gamedev.stackexchange.com/questions/17620/equation-to-make-small-number-big-and-big-number-small-gravity
        self.dist_fitness = (300 / (self.dist + .1))
        self.time_fitness = (300 / (time + .1))
        # use prioritized fitness algorithm
        # Source : https://geekyisawesome.blogspot.com/2013/06/fitness-function-for-multi-objective.html
        # print("Fitness of Dist: %.2f\t Time: %.2f" % (self.dist_fitness, self.time_fitness))
        self.fitness = a * b * (self.dist_fitness * 2 * self.time_fitness)

    def draw(self, screen):
        screen.blit(self.picture.image, self.picture.rect)


class Obstacle:
    def __init__(self, topleft, bottomright):
        width_hieght = bottomright - topleft
        width, height = width_hieght.x, width_hieght.y
        self.rect = pygame.Rect(topleft.x, topleft.y, width, height)
        self.color = (105, 105, 105)
        self.hover = False
        if width < 0:
            width = topleft.x - bottomright.x
            self.rect.width = width
            self.rect.left = topleft.x - width
        if height < 0:
            height = topleft.y - bottomright.y
            self.rect.height = height
            self.rect.top = topleft.y - height

    def handle_event(self, event):
        if self.rect.collidepoint(pygame.mouse.get_pos()):
            print(pygame.mouse.get_pos())
            if not self.hover:
                print('setting to hover color')
                self.color = (145, 105, 105)
                self.hover = True
        else:
            self.color = (105, 105, 105)
            self.hover = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                return True

    def draw(self, screen, color=None):
        if color is not None:
            pygame.draw.rect(screen, color, self.rect)
        else:
            pygame.draw.rect(screen, self.color, self.rect)

    def collide(self, other):
        if self.rect.colliderect(other.rect):
            return True
