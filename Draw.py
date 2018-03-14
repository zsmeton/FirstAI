# Main Looping File
# Sets up population
# Draws the current generation

import random

# # Import Libraries # #
import pygame

from GeneticAlgorithm import Group, Settings, Statist, Target

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
generation = 1
average_fitness = 0

# # Drawing Options # #
debug = False
new_population = False
draw = True
fast_forward = False

# population
population = Group.Population(100)
obstacles = Group.Obstacles()

# target
Target.__init__()

while setup:
    population.create_population()
    setup = False
    continue


while drawing:
    Settings.timer()
    for event in pygame.event.get():
        if draw:
            obstacles.handle_event(event)
        if event.type == pygame.QUIT:
            # Before the simulation exists exits
            # run stats.txt
            Statist.run_stats(generation, population)
            Statist.generate_graph(generation)
            # save stats.txt to a file
            pygame.quit()
            quit(0)
            pygame.quit()
        elif event.type == pygame.KEYUP:
            # exits code
            if event.key == pygame.K_ESCAPE:
                # Before the simulation exists exits
                # run stats.txt
                Statist.run_stats(generation, population)
                Statist.generate_graph(generation)
                # save stats.txt to a file
                pygame.quit()
                quit(0)
            elif event.key == pygame.K_z:
                debug = not debug
                draw = False
            elif event.key == pygame.K_d:
                draw = not draw
                fast_forward = False
            elif event.key == pygame.K_f:
                fast_forward = not fast_forward
                draw = False
            elif event.key == pygame.K_g:
                Statist.run_stats(generation, population)
                Statist.generate_graph(generation)

    population.update(obstacles)

    if fast_forward:
        if random.randint(0, 20) is 1:
            # Fills screen with white
            screen.fill(Settings.WHITE)
            population.draw(screen)
            obstacles.draw(screen)
            Target.draw(screen)
            time_text = "Time: " + str(round(Settings.time))
            time_draw = myfont.render(time_text, 1, (0, 0, 0))
            screen.blit(time_draw, [10, 10])
    else:
        screen.fill(Settings.WHITE)
        population.draw(screen)
        obstacles.draw(screen)
        Target.draw(screen)
        time_text = "Time: " + str(round(Settings.time))
        time_draw = myfont.render(time_text, 1, (0, 0, 0))
        screen.blit(time_draw, [10, 10])

    time_text = "Generation: " + str(round(generation))
    time_draw = myfont.render(time_text, 1, (0, 0, 0))
    screen.blit(time_draw, [10, 22])

    # if debug is active show the vector field of the best rocket
    if debug:
        population.debug(screen)

    if Settings.time > Settings.max_time:
        new_population = True

    if new_population:
        Settings.new_rand()
        population.calculate_fitness()
        Statist.run_stats(generation, population)
        population.selection()
        population.reproduction()
        Settings.timer(reset=True)
        generation += 1
        new_population = False

    # Draws everything
    pygame.display.flip()

    if draw:
        clock.tick(20)
    else:
        clock.tick(2000000)

pygame.quit()
