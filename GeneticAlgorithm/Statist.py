# Collection of statistics functions and stat features
import numpy as np
from matplotlib import pyplot as plt

average_fitness = []
best_fitness = []
number_of_hits = []
best_time = []


class Graph:
    def __init__(self, x, **kwargs):
        self.x_values = np.linspace(0, x, x)
        self.labels = []
        self.y_values = []
        if kwargs is not None:
            for key, value in kwargs.items():
                self.labels.append(key)
                self.y_values.append(value)

    def draw(self):
        fig, ax = plt.subplots()
        plt.xlabel('Generations')
        for i, plot in enumerate(self.labels):
            print(self.x_values, self.y_values[i])
            ax.scatter(self.x_values, self.y_values[i], label=str(plot))
        ax.legend()
        ax.grid(True)
        plt.show()


def run_stats(generation, population):
    # add data to lists
    average_fitness.append(population.average_fitness)
    best_fitness.append(population.best_fitness)
    best_time.append(population.best_object.hit_time)
    print('Gen: %d\tAvg: %.2f\tBest: %.2f\tTime: %d' % (generation, average_fitness[-1], best_fitness[-1], best_time[-1]))


def variable_mapping(value, from_low, from_high, to_low, to_high):
    old_range = (from_high - from_low)
    new_range = (to_high - to_low)
    new_value = (((value - from_low) * new_range) / old_range) + to_low
    return new_value


def generate_graph(generation):
    graph = Graph(generation, avg_fitness=average_fitness, best_time=best_time, best_fitness=best_fitness)
    graph.draw()
