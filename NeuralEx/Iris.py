import csv
import importlib
import random

from PyZ import Neural

importlib.reload(Neural)


def predict(output):
    ind = 0
    maximum = output[0]
    for j, num in enumerate(output):
        if num > maximum:
            ind = j
    prediction = ind
    if prediction == 0:
        return "Iris-setosa"
    elif prediction == 1:
        return "Iris-versicolor"
    elif prediction == 2:
        return "Iris-virginica"
    else:
        print("WHAT THE FUCK, RICHARD")


inputs = []
answer = []

with open("iris.csv", "r") as input_file:
    line_reader = csv.reader(input_file, delimiter=',')
    for line in line_reader:
        data = []
        for cell in line:
            data.append(cell)
        inputs.append([float(x) for x in data[:-1]])
        if data[-1] == "Iris-setosa":
            answer.append([1, 0, 0])
        elif data[-1] == "Iris-versicolor":
            answer.append([0, 1, 0])
        elif data[-1] == "Iris-virginica":
            answer.append([0, 0, 1])
        else:
            print("AHHH What the fucking hell is this bullshit")

nn = Neural.NeuralNetwork(input_nodes=4, hidden_nodes=8, hidden_layers=4, output_nodes=3)
nn.draw()

for i in range(100):
    index = random.randint(0, 149)
    print("Iteration", i, ":")
    print("Input:", inputs[index], "Target:", answer[index])
    nn.train(inputs[index], answer[index])

for i in range(30):
    index = random.randint(0, 149)
    print("Output:", predict(nn.feedforward(inputs[index])))
    print("Correct:", predict(answer[index]))
    print()
