# # Imports # #
from scipy.special import expit

from GeneticAlgorithm import DNA, Vector, Graphics, Settings, Target, Statist


# Main object which is being optimized
class Rocket:
    def __init__(self, DNA_=None):
        self.position = Vector.Vector(10, Settings.height/2)
        self.velocity = Vector.Vector()
        self.acceleration = Vector.Vector()
        self.DNA = DNA.DNA(random_=True)
        if DNA_ is not None:
            self.DNA = DNA_
        self.picture = Graphics.Image('rocket.png', [self.position.x, self.position.y])
        self.fitness = 0
        self.hit_target = False
        self.hit_time = Settings.max_time
        self.time_fitness = 0
        self.dist = Settings.width
        self.dist_fitness = 0

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
            self.hit_time = Settings.time
            self.hit_target = True
            return False, True
        else:
            return True, False

    def update_fitness(self):
        self.dist = Target.is_reached(self)
        if self.hit_target:
            a = 2
        else:
            a = 1
        # map time to same value as distance
        time = Statist.variable_mapping(self.hit_time, Settings.min_time, Settings.max_time, Target.size, Settings.width)
        # find the fitness of each using inverse
        # Source : https://gamedev.stackexchange.com/questions/17620/equation-to-make-small-number-big-and-big-number-small-gravity
        self.dist_fitness = (300 / (self.dist + .1))
        self.time_fitness = (300 / (time + .1))
        # use prioritized fitness algorithm
        # Source : https://geekyisawesome.blogspot.com/2013/06/fitness-function-for-multi-objective.html
        # print("Fitness of Dist: %.2f\t Time: %.2f" % (self.dist_fitness, self.time_fitness))
        self.fitness = a * (self.dist_fitness + expit(self.time_fitness))

    def draw(self, screen):
        screen.blit(self.picture.image, self.picture.rect)


