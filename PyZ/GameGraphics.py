import pygame
from pygame import gfxdraw as gfx


def draw_circle(surface, color, pos, radius, aa=False):
    """Draws a circle on the current pygame screen
        Arguments:
            surface: the surface to which the circle should be drawn on
            color: (tuple, list) the rgb colors eg.(255,10,33)
            pos: (tuple, list) the x and y coordinate of the circle can be float
            radius: the radius of the circle
            aa: if true a antialiased circle will be drawn
    """
    pos_x = round(pos[0])
    pos_y = round(pos[1])
    pos = [round(pos[0]),round(pos[1])]
    if aa:
        gfx.aacircle(surface, pos_x, pos_y, radius, color)
        gfx.filled_circle(surface, pos_x, pos_y, radius, color)
    else:
        pygame.draw.circle(surface, color, pos, radius)


# Source : http://www.andrewnoske.com/wiki/Code_-_heatmaps_and_color_gradients
def color_gradient(value, start=(255,255,255), end=(0,0,0)):
    """Returns a tuple of the RGB values of the point on a gradient
        Arguments:
            value: A float between 0 and 1 which will mapped onto a gradient
            start: the low color of the gradient (default red)
            end: the high color of the gradient (default blue)
    """
    a_r, a_g, a_b = start
    b_r, b_g, b_b = end

    red = round((b_r - a_r) * value + a_r)
    green = round((b_g - a_g) * value + a_g)
    blue = round((b_b - a_b) * value + a_b)
    return (red,green,blue)

