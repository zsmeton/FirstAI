# # Imports # #
import random

import pygame

from GeneticAlgorithm import Individuals, Target, Vector

pygame.init()
random.seed()


class Population:
    def __init__(self, size_):
        self.size = size_
        self.population_objects = []
        self.alive_population = []
        self.mating_pool = []
        self.mutationRate = 0.1
        self.best_object = None
        self.best_fitness = 0
        self.average_fitness = 0

    # Creates a randomly generated first population
    def create_population(self):
        for pop in range(self.size):
            current_rocket = Individuals.Rocket()
            self.population_objects.append(current_rocket)
            self.alive_population.append(current_rocket)

    def update(self, current_obstacles):
        for rocket in self.alive_population:
            rocket.update()
            rocket.collision(current_obstacles)
            if not rocket.alive:
                self.alive_population.remove(rocket)

    def draw(self, screen):
        for rocket in self.population_objects:
            rocket.draw(screen)

    def debug(self, screen):
        self.calculate_fitness()
        pygame.draw.circle(screen, (105, 105, 250), (int(self.best_object.position.x), int(self.best_object.position.y)), 5, 1)
        pygame.draw.circle(screen, (105, 105, 250), (int(Target.position.x), int(Target.position.y)), 5, 1)
        self.best_object.DNA.debug(screen)

    def calculate_fitness(self):
        self.best_fitness = 0
        self.average_fitness = 0
        for rocket in self.population_objects:
            rocket.update_fitness()
            self.average_fitness += rocket.fitness
            if rocket.fitness > self.best_fitness:
                self.best_fitness = rocket.fitness
                self.best_object = rocket
        self.average_fitness /= len(self.population_objects)
        return self.average_fitness

    def selection(self):
        self.mating_pool.clear()
        for rocket in self.population_objects:
            fitness = rocket.fitness / self.best_fitness
            number_in_pool = fitness * 100
            for i in range(int(round(number_in_pool))):
                self.mating_pool.append(rocket)
        print(self.mating_pool.count(self.best_object)/len(self.mating_pool))

    def reproduction(self):
        # refill population with new generation
        self.alive_population.clear()
        self.population_objects.clear()
        for i in range(self.size):
            mom = self.mating_pool[random.randint(0, len(self.mating_pool) - 1)]
            dad = self.mating_pool[random.randint(0, len(self.mating_pool) - 1)]
            mom_dna = mom.DNA
            dad_dna = dad.DNA
            child = mom_dna.cross_over(dad_dna, mutation_rate=0.035)
            child_rocket = Individuals.Rocket(child)
            self.population_objects.append(child_rocket)
            self.alive_population.append(child_rocket)


class Obstacles:
    def __init__(self):
        self.obstacle_list = []
        self.obstacle_chosen = False
        self.dragging = False
        self.current_corner = Vector.Vector()
        self.drag_object = None

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.obstacle_chosen = False
            print("I gotta click")
            for obstacle in self.obstacle_list:
                if obstacle.handle_event(event):
                    self.obstacle_list.remove(obstacle)
                    self.obstacle_chosen = True
                    self.dragging = False
            if not self.obstacle_chosen:
                if not self.dragging:
                    print("Im gonna build  ")
                    x, y = event.pos
                    self.current_corner.set(x_=x, y_=y)
                    self.dragging = True
                else:
                    x, y = event.pos
                    new_corner = Vector.Vector(x_=x, y_=y)
                    if new_corner.dist(self.current_corner) <= 20:
                        self.dragging = False
                        self.obstacle_chosen = False
                    else:
                        self.dragging = False
                        new = Individuals.Obstacle(topleft=self.current_corner, bottomright=new_corner)
                        self.obstacle_list.append(new)

    def draw(self, screen):
        if self.dragging and not self.obstacle_chosen:
            x, y = pygame.mouse.get_pos()
            temp = Vector.Vector(x, y)
            self.drag_object = Individuals.Obstacle(self.current_corner, temp)
            self.drag_object.draw(screen, (200, 200, 200))
        for obstacle in self.obstacle_list:
            obstacle.draw(screen)

    def collide(self, other):
        for obstacle in self.obstacle_list:
            if obstacle.collide(other):
                return True
