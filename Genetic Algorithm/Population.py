# # Imports # #
import random

import pygame

from GeneticAlgorithm import Rocket

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

    # Creates a randomly generated first population
    def create_population(self, width_, height_):
        for pop in range(self.size):
            temp_object = Rocket.Rocket(width_, height_)
            self.population_objects.append(temp_object)
            self.alive_population.append(temp_object)

    def draw(self, screen):
        for rocket in self.alive_population:
            alive, hit = rocket.update()
            if alive:
                screen.blit(rocket.picture.image, rocket.picture.rect)
            elif not alive and hit:
                print("----HIT----")
                self.alive_population.remove(rocket)
                self.breeding_population.append(rocket)
            else:
                self.alive_population.remove(rocket)

    def best_object(self):
        self.alive_population[0].update_fitness()
        best_fitness = self.alive_population[0].fitness
        best = self.alive_population[0]
        for rocket in self.alive_population:
            rocket.update_fitness()
            if rocket.fitness > best_fitness:
                best_fitness = rocket.fitness
                best = rocket
        return best

    def best_fitness(self):
        self.alive_population[0].update_fitness()
        best_fitness = self.alive_population[0].fitness
        best = self.alive_population[0]
        for rocket in self.alive_population:
            rocket.update_fitness()
            if rocket.fitness > best_fitness:
                best_fitness = rocket.fitness
                best = rocket
        return best_fitness

    def average(self):
        average_fitness = 0
        for rocket in self.alive_population:
            rocket.update_fitness()
            average_fitness += rocket.fitness
        average_fitness /= len(self.alive_population)
        return average_fitness

    def debug(self, screen):
        best = self.best_object()
        pygame.draw.circle(screen, (105, 105, 250), (int(best.position.x), int(best.position.y)), 5, 1)
        pygame.draw.circle(screen, (105, 105, 250), (int(best.target.x), int(best.target.y)), 5, 1)
        best.DNA.debug(screen)

    def selection(self):
        self.alive_population += self.breeding_population
        self.mating_pool = []
        best_fitness = self.best_fitness()
        for rocket in self.alive_population:
            fitness = rocket.fitness / best_fitness
            number_in_pool = fitness * 100
            for i in range(int(number_in_pool)):
                self.mating_pool.append(rocket)

    def reproduction(self):
        # refill population with new generation
        self.alive_population = []
        self.population_objects = []
        for i in range(self.size):
            mom = self.mating_pool[random.randint(0, len(self.mating_pool) - 1)]
            dad = self.mating_pool[random.randint(0, len(self.mating_pool) - 1)]
            width = mom.width
            height = mom.width
            mom_dna = mom.DNA
            dad_dna = dad.DNA
            child = mom_dna.cross_over(dad_dna)
            child_rocket = Rocket.Rocket(width, height, child)
            self.population_objects.append(child_rocket)
            self.alive_population.append(child_rocket)
