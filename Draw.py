# Main Looping File
# Sets up population
# Draws the current generation

# # Import Libraries # #
import pygame
from GeneticAlgorithm import Group, Graphics, Settings, Individuals, Target

# # Constants # #
Settings.__init__()

# # Pygame Setup # #
pygame.init()
screen = pygame.display.set_mode([Settings.width, Settings.height])
pygame.display.set_caption("Genetic Path Finding")  # name of the window created
clock = pygame.time.Clock()  # used to manage how fast the screen updates
myfont = pygame.font.Font(None, 12)  # sets the font for text in pygame


# Game state
setup = True
drawing = True

# statistics
timer = 0
generation = 1
average_fitness = 0
fitness = []

# # Drawing Options # #
debug = False
new_population = False

# population
population = Group.Population(100)

# target
Target.__init__()

while setup:
    population.create_population()
    setup = False
    continue


while drawing:
    timer += 1
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
        elif event.type == pygame.KEYUP:
            # exits code
            if event.key == pygame.K_ESCAPE:
                pygame.quit()
                quit(0)
            elif event.key == pygame.K_d:
                debug = not debug
            elif event.key == pygame.K_r:
                new_population = True
            elif event.key == pygame.K_g:
                graph = Graphics.Graph(generation, fitness)
                graph.draw()

    # Fills screen with white
    screen.fill(Settings.WHITE)

    population.update()
    population.draw(screen)

    Target.draw(screen)

    time_text = "Time: " + str(round(timer))
    time_draw = myfont.render(time_text, 1, (0, 0, 0))
    screen.blit(time_draw, [10, 10])
    time_text = "Generation: " + str(round(generation))
    time_draw = myfont.render(time_text, 1, (0, 0, 0))
    screen.blit(time_draw, [10, 22])
    average_fitness = population.average()
    time_text = "Average Fitness: " + str(average_fitness)
    time_draw = myfont.render(time_text, 1, (0, 0, 0))
    screen.blit(time_draw, [10, 34])

    # if debug is active show the vector field of the best rocket
    if debug:
        population.debug(screen)
    if timer > 600:
        new_population = True

    if new_population:
        average_fitness = population.average()
        fitness.append(average_fitness)
        population.selection()
        population.reproduction()
        new_population = False
        timer = 0
        generation += 1
        print(fitness[-1])

    # Draws everything
    pygame.display.flip()
    clock.tick(6000)

pygame.quit()
