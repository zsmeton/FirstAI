import math

import numpy as np
import pygame

from PyZ import GameGraphics as gg

pygame.init()


class NeuralNetwork:
    np.random.seed()

    def __init__(self, input_nodes, hidden_nodes, hidden_layers, output_nodes):
        """A neural network class which generates a neural network
           Arguments:
                input_nodes: (int) number of inputs into the network
                hidden_nodes: (int) number of hidden nodes to do the calculations
                hidden_layers: (int) the number of hidden node layers
                output_nodes: (int) the amount of possible decisions the layer can make
        """
        # Class members:
        # num_input_nodes
        # num_hidden_nodes
        # num_hidden_layers
        # num_output_nodes
        # weights = [[num_hidden_nodes, num_input_nodes],[num_hidden_nodes, num_hidden_nodes],[]<- for each hl,
        # [num_output_nodes, num_hidden_nodes]]
        # biases

        self.num_input_nodes = input_nodes
        self.num_hidden_nodes = hidden_nodes
        self.num_hidden_layers = hidden_layers
        self.num_output_nodes = output_nodes

        self.weights = []
        for i in range(self.num_hidden_layers + 1):
            if i is 0:
                # first weights array is input to hidden
                self.weights.append(2 * np.random.rand(self.num_hidden_nodes, self.num_input_nodes) - 1)

            elif i < self.num_hidden_layers:
                # next weight array is hidden nodes to hidden nodes
                self.weights.append(2 * np.random.rand(self.num_hidden_nodes, self.num_hidden_nodes) - 1)
            else:
                # last weight array is hidden nodes to output nodes
                self.weights.append(2 * np.random.rand(self.num_output_nodes, self.num_hidden_nodes) - 1)

        self.biases = []
        for i in range(self.num_hidden_layers + 1):
            if i < self.num_hidden_layers:
                # for every hidden node there is a bias
                self.biases.append(2 * np.random.rand(self.num_hidden_nodes) - 1)
            else:
                # for the output node there is a bias as well
                self.biases.append(2 * np.random.rand(self.num_output_nodes) - 1)

        self.activation = np.vectorize(self.tanh, otypes=[float])

    @staticmethod
    def tanh(value):
        return (1 - math.exp(-2 * value)) / (1 + math.exp(-2 * value))

    def node_output(self, inputs, weights, biases):
        node_output = weights.dot(inputs)
        node_output = np.add(node_output, biases)
        node_output = self.activation(node_output)
        return node_output

    def feed_forward(self, inputs):
        """Returns the output of the neural network given an input"""
        hidden_output = self.node_output(inputs, self.weights[0], self.biases[0])
        print(hidden_output)
        for layer in range(1, self.num_hidden_layers-1):
            hidden_output = self.node_output(hidden_output, self.weights[layer], self.biases[layer])
            print(hidden_output)

        hidden_output = self.node_output(hidden_output, self.weights[-1], self.biases[-1])
        print(hidden_output)
        return hidden_output

    def train(self, inputs, answers):
        output = self.feed_forward(inputs)
        error = np.subtract(answers, output)
        weights_t = []
        hidden_errors = []
        for weight in self.weights:
            weights_t.append(np.transpose(weight))
        for weight in weights_t:
            hidden_errors = weight.dot(error)

    def node_pos(self, spacing, type, layer, node):
        """Returns the x and y position of the node
            Arguments:
                spacing: the desired space at the top and bottom
                type: either input, hidden or output
                layer: the current layer in the network input and output just use 1
                node: the node position coming top down 0 - __"""
        width = (self.num_hidden_layers + 3) * spacing * 2
        values = [self.num_input_nodes, self.num_hidden_nodes, self.num_output_nodes]
        values.sort(reverse=True)
        height = (values[0] + 1) * spacing
        h_percentile = height - spacing
        w_percentile = (width - (spacing * 2)) / (self.num_hidden_layers + 2)
        if type == 'input':
            pos_y = h_percentile / (self.num_input_nodes + 1)
            pos_y *= (node + 1)
            pos_x = w_percentile
            return (pos_x, pos_y)
        elif type == 'hidden':
            pos_y = h_percentile / (self.num_hidden_nodes + 1)
            pos_y *= (node + 1)
            pos_x = w_percentile * (layer + 2)
            return (pos_x, pos_y)
        elif type == 'output':
            pos_y = h_percentile / (self.num_output_nodes + 1)
            pos_y *= (node + 1)
            pos_x = w_percentile * (self.num_hidden_layers + 2)
            return (pos_x, pos_y)
        else:
            print("Invalid argument: type")
            return 1

    def draw(self):
        """Draws the neural network in a new pygame window with node bias and connections weights shown"""
        spacing = 50
        # # Pygame Setup # #
        # calculate how wide and tall it needs to be
        width = (self.num_hidden_layers + 3) * spacing * 2
        values = [self.num_input_nodes, self.num_hidden_nodes, self.num_output_nodes]
        values.sort(reverse=True)
        height = (values[0] + 1) * spacing
        pygame.init()
        screen = pygame.display.set_mode([width, height])
        pygame.display.set_caption("Genetic Path Finding")  # name of the window created
        clock = pygame.time.Clock()  # used to manage how fast the screen updates
        myfont = pygame.font.Font(None, 12)  # sets the font for text in pygame
        drawing = True
        while drawing:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
            screen.fill((255, 255, 255))

            h_percentile = height - spacing
            w_percentile = (width - (spacing * 2)) / (self.num_hidden_layers + 2)
            # Nodes
            for node in range(self.num_input_nodes):
                pos = h_percentile / (self.num_input_nodes + 1)
                gg.draw_circle(screen, (105, 105, 105), self.node_pos(spacing, 'input', 1, node), 5, aa=True)
            for layer in range(self.num_hidden_layers):
                for node in range(self.num_hidden_nodes):
                    pos = h_percentile / (self.num_hidden_nodes + 1)
                    bias = self.biases[layer][node]
                    color = gg.color_gradient(bias)
                    gg.draw_circle(screen, color, self.node_pos(spacing, 'hidden', layer, node), 5, aa=True)
            for node in range(self.num_output_nodes):
                pos = h_percentile / (self.num_output_nodes + 1)
                bias = self.biases[-1][node]
                color = gg.color_gradient(bias)
                gg.draw_circle(screen, color, self.node_pos(spacing, 'output', 1, node), 5, aa=True)

            # Connections
            for inp in range(self.num_input_nodes):
                for node in range(self.num_hidden_nodes):
                    weight = self.weights[0][node][inp]
                    color = gg.color_gradient(weight)
                    pygame.draw.aaline(screen, color, self.node_pos(spacing, 'input', 1, inp),
                                       self.node_pos(spacing, 'hidden', 0, node))
            for layer in range(0, self.num_hidden_layers-1):
                for node in range(self.num_hidden_nodes):
                    for other in range(self.num_hidden_nodes):
                        weight = self.weights[layer+1][other][node]
                        color = gg.color_gradient(weight)
                        pygame.draw.aaline(screen, color, self.node_pos(spacing, 'hidden', layer, node),
                                           self.node_pos(spacing, 'hidden', layer + 1, other))
            for node in range(self.num_hidden_nodes):
                for out in range(self.num_output_nodes):
                    layer = self.num_hidden_layers
                    weight = self.weights[layer][out][node]
                    color = gg.color_gradient(weight)
                    pygame.draw.aaline(screen, color, self.node_pos(spacing, 'hidden', layer-1, node),
                                       self.node_pos(spacing, 'output', 1, out))

            pygame.display.flip()
            clock.tick(20)


if __name__ == '__main__':
    print('DEBUGGING:')
    a = NeuralNetwork(input_nodes=4, hidden_nodes=3, hidden_layers=2, output_nodes=1)
    data = [1, 1, .2, .3]
    a.feed_forward(data)
    a.draw()
