import csv
import importlib
import random

import numpy as np

from PyZ import Neural

importlib.reload(Neural)


class Switch:
    def __init__(self, size):
        self.items = [0] * size

    def turnOn(self, index):
        self.items = [0 for x in self.items]
        self.items[index] = 1

    def turnList(self, list):
        ind = 0
        maximum = list[0]
        for j, num in enumerate(list):
            if num > maximum:
                ind = j
        prediction = ind
        self.turnOn(prediction)

    def print(self):
        ind = 0
        maximum = self.items[0]
        for j, num in enumerate(self.items):
            if num > maximum:
                ind = j
        print(ind)

    def __eq__(self, other):
        if self.items == other:
            return True
        else:
            return False


inputs = []
answer = []
classifications = 29

# Read in data set
with open("Abalone.csv", "r") as input_file:
    line_reader = csv.reader(input_file, delimiter=',')
    for line in line_reader:
        data = []
        for i, cell in enumerate(line):
            # turn F, M, I into numbers
            if i is 0:
                if cell == 'M':
                    data.append(1)
                elif cell == 'F':
                    data.append(-1)
                elif cell == 'I':
                    data.append(0)
                else:
                    print("Value Error: cell cannot be converted to int")
            else:
                data.append(cell)

        inputs.append([float(x) for x in data[:-1]])
        ans = Switch(classifications)
        ans.turnOn(int(data[-1]) - 1)
        answer.append(ans.items)

# store best neural network
best = Neural.NeuralNetwork(input_nodes=8, hidden_nodes=19, hidden_layers=1, output_nodes=29)
# store lowest error percent
error = 200
# store the rate of the best nn
best_rate = 0

# Change learning rate
for rate in np.arange(0.01, 0.03, 0.005):
    # Change number of hidden nodes:
    for node_num in range(15, 24):
        # Change number of hidden layers:
        for num_layer in range(1, 7):
            print("Testing with rate(%.2f), hidden nodes(%d), hidden layers(%d)" % (rate, node_num, num_layer))
            # Initialize:
            nn = Neural.NeuralNetwork(input_nodes=8, hidden_nodes=node_num, hidden_layers=num_layer, output_nodes=29)
            # Train:
            for i in range(int(3 * len(inputs) / 4)):
                index = random.randint(0, len(inputs) - 1)
                nn.train(inputs[index], answer[index], 0.01)

            # Verify:
            output = Switch(classifications)
            expected = Switch(classifications)
            correct = 0
            total = 0
            for i in range(int(len(inputs) / 4)):
                total = total + 1
                index = random.randint(0, len(inputs) - 1)

                output.turnList(nn.feedforward(inputs[index]))
                if answer[index] == output.items:
                    correct = correct + 1
            cur_error = (1 - (correct / total)) * 100
            print("Error:", cur_error, "%")
            print()

            # Store best
            if cur_error < error:
                best = nn
                error = cur_error
                best_rate = rate

print("Best Network had rate(%.3f), hidden nodes(%d), hidden layers(%d)" % (
best_rate, best.num_hidden_nodes, best.num_hidden_layers))
print("With Error:", error, "%")
best.draw()
