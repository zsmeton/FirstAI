# # Imports # #
import random
import pygame
from GeneticAlgorithm import Individuals, Target

pygame.init()
random.seed()


class Population:
    def __init__(self, size_):
        self.size = size_
        self.population_objects = []
        self.alive_population = []
        self.breeding_population = []
        self.mating_pool = []
        self.mutationRate = 0.1
        self.best_object = None
        self.best_fitness = 0
        self.average_fitness = 0

    # Creates a randomly generated first population
    def create_population(self):
        for pop in range(self.size):
            temp_object = Individuals.Rocket()
            self.population_objects.append(temp_object)
            self.alive_population.append(temp_object)

    def update(self):
        for rocket in self.alive_population:
            alive, hit = rocket.update()
            if not alive and hit:
                self.alive_population.remove(rocket)
                self.breeding_population.append(rocket)
            elif not alive:
                self.alive_population.remove(rocket)

    def draw(self, screen):
        for rocket in self.alive_population:
            rocket.draw(screen)
        for rocket in self.breeding_population:
            rocket.draw(screen)

    def debug(self, screen):
        self.calculate_fitness()
        pygame.draw.circle(screen, (105, 105, 250), (int(self.best_object.position.x), int(self.best_object.position.y)), 5, 1)
        pygame.draw.circle(screen, (105, 105, 250), (int(Target.position.x), int(Target.position.y)), 5, 1)
        self.best_object.DNA.debug(screen)

    def calculate_fitness(self):
        self.best_fitness = 0
        for rocket in self.population_objects:
            rocket.update_fitness()
            self.average_fitness += rocket.fitness
            if rocket.fitness > self.best_fitness:
                self.best_fitness = rocket.fitness
                self.best_object = rocket
        self.average_fitness /= len(self.population_objects)
        return self.average_fitness

    def selection(self):
        self.alive_population += self.breeding_population
        self.mating_pool = []
        for rocket in self.alive_population:
            fitness = rocket.fitness / self.best_fitness
            number_in_pool = fitness * 100
            for i in range(int(number_in_pool)):
                self.mating_pool.append(rocket)

    def reproduction(self):
        # refill population with new generation
        self.breeding_population = []
        self.alive_population = []
        self.population_objects = []
        for i in range(self.size):
            mom = self.mating_pool[random.randint(0, len(self.mating_pool) - 1)]
            dad = self.mating_pool[random.randint(0, len(self.mating_pool) - 1)]
            mom_dna = mom.DNA
            dad_dna = dad.DNA
            child = mom_dna.cross_over(dad_dna, mutation_rate=0.02)
            child_rocket = Individuals.Rocket(child)
            self.population_objects.append(child_rocket)
            self.alive_population.append(child_rocket)

