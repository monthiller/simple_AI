# Imports
from .neuron import Neuron
import matplotlib.pyplot as plt
import math
import random
import numpy as np
from scipy.special import expit
import networkx as nx

class Network:
  def __init__(self, name, **kwargs):
    self.neurons = set()
    self.input_layer = list()
    self.output_layer = list()
    self.inputs = None
    self._target = None
    self.learning_rate = 0.1
    self.errors = list()
    self._neuron_dict = {}
    self.ready = False
    self.name = name
    self.predicted = None
    self.training = kwargs["training"]
    self.exercice_errors = []
    self.connections = []

    self._init_neurons(*kwargs["neurons"])
    self._init_inputs(*kwargs["inputs"])
    self._init_outputs(*kwargs["outputs"])

  def _init_neurons(self, *args):
    for params in args:
      neuron = Neuron(**params)
      self.add_neuron(neuron)

  def _init_inputs(self, *args):
    for name in args:
      self.input_layer.append(self._neuron_dict[name])

  def _init_outputs(self, *args):
    for name in args:
      self.output_layer.append(self._neuron_dict[name])

  def _init_connections(self, *args):
    for origin, destination in args:
      self.connect(origin, destination)

  def __str__(self):
    return f"network {self.name}"

  def __getitem__(self, key):
    return self._neuron_dict[key]

  def add_neuron(self, neuron):
    if neuron.name in self._neuron_dict:
      assert neuron is self._neuron_dict[neuron.name]
      return
    self._neuron_dict[neuron.name] = neuron

  def new(self):
    copy = Network()
    copy_neurons = []
    for neuron in self.neurons:
      _neuron = Neuron(name=neuron.name)
      copy.add_neuron(_neuron)

    for neuron in self.input_layer:
      copy.add_input(copy[neuron.name])

    for neuron in self.output_layer:
      copy.add_output(copy[neuron.name])

    for neuron1, neuron2 in self.connections:
      copy.connect(
          copy[neuron1.name],
          copy[neuron2.name]
          )

    return copy

  def copy(self):
    raise NotImplementedError("Implement Neuron.copy first!")
    copy = Network()
    copy_neurons = []
    for neuron in self.neurons:
      copy.add_neuron(neuron.copy())

    for neuron in self.input_layer:
      copy.add_input(copy[neuron.name])

    for neuron in self.output_layer:
      copy.add_output(copy[neuron.name])

    for neuron1, neuron2 in self.connections:
      copy.connect(neuron1, neuron2)

    return copy

  def predict(self, inputs):
    assert len(inputs) == len(self.input_layer), f"{inputs=}, {self.input_layer=}"
    for input, neuron in zip(inputs, self.input_layer):
      neuron.inputs = [input]
    # output will call previous neuron for prediction
    self.predicted = [output.predict() for output in self.output_layer]
    return self.predicted

  def add_input(self, neuron):
    self.add_neuron(neuron)
    self.input_layer.append(neuron)
    if not neuron.weights:
      neuron.weights.append(1)
    self.neurons.add(neuron)

  def add_output(self, neuron):
    self.add_neuron(neuron)
    self.output_layer.append(neuron)
    self.neurons.add(neuron)

  def connect(self, neuron1, neuron2):
    if neuron1.name not in self._neuron_dict:
      self.add_neuron(neuron1)
    assert neuron2 in self.neurons
    neuron1.output_connections.append(neuron2)
    neuron2.input_connections.append(neuron1)
    neuron2.weights.append(1)
    self.neurons.add(neuron1)
    self.connections.append((neuron1, neuron2))

  def error(self, target):
      error = sum([(p-t)**2 for p, t in zip(self.predicted, target)])
      return error/len(target)

  def derr_w(self, neuron, i):
    derrors = []
    for output, t in zip(self.output_layer, self._target):
      derrors.append(
          2*(output.predict() - t)*output.dp_w(neuron, i)
      )

    return sum(derrors)

  def derr_b(self, neuron):
    derrors = []
    for output, t in zip(self.output_layer, self._target):
      derrors.append(
          2*(output.predict() - t)*output.dp_b(neuron)
      )

    return sum(derrors)

  def correct(self):
    target = self._target

    weight_corrections = {}
    bias_corrections = {}

    for neuron in sorted(self.neurons, key=lambda x:x.name):
      c_w = []
      for i, _ in enumerate(neuron.weights):
        c_w.append(self.derr_w(neuron, i))

      weight_corrections[neuron]=c_w

      bias_corrections[neuron]=self.derr_b(neuron)

    for neuron in self.neurons:
      c_w = weight_corrections[neuron]
      for i, _ in enumerate(neuron.weights):
        neuron.weights[i] -= c_w[i] * self.learning_rate
      neuron.bias -= bias_corrections[neuron] * self.learning_rate

  def exercise(self, inputs, target):
    # predict result to define the error
    self.predict(inputs)
    for input, neuron in zip(inputs, self.input_layer):
      neuron.inputs = [input]
    self._target = target
    self.exercice_errors.append(self.error(target))
    # correct the weights and bias
    self.correct()

  def train(self, repetitions):
    training = self.training
    inputs = training["inputs"]
    targets = training["outputs"]
    tolerances = training["tolerance"]

    x = []
    y = []
    for i in range(repetitions):
      self.exercice_errors.clear()
      for input, target in zip(inputs, targets):
          self.exercise(input, target)

      self.errors.append(sum(self.exercice_errors)/len(self.exercice_errors))

  def test(self, check):
    inputs = check["inputs"]
    targets = check["outputs"]
    tolerances = check["tolerance"]
    achieved = list()
    for input, target, tolerance in zip(inputs, targets, tolerances, strict=True):
      output = self.predict(input)[0]
      achieved.append(tolerance[0]<= output <= tolerance[1])

    if all(achieved):
      self.ready = True
      print(f"Tolerance achieved with {len(self.errors)} total sessions!")

  def random_fit(self, seed=None):
    if seed:
      random.seed(seed)

    for neuron in self.neurons:
      for i, w in enumerate(neuron.weights):
        neuron.weights[i] = random.uniform(-50, 50)

      neuron.bias = random.uniform(-50, 50)

  def plot(self):
    fig, axs = plt.subplots(nrows=2, ncols=1)
    axs[0].set_title(f"Trained for {len(self.errors)} repetitions, with {self.learning_rate} learning rate.")
    axs[0].grid(True)

    x = [x[0] for x in self.training["inputs"]]
    y = [y[0] for y in self.training["outputs"]]
    axs[0].plot(x, y, label='target')

    x = [i/10 for i in range(-10, 250)]
    y = [self.predict([i]) for i in x]
    print(max(y))
    axs[0].plot(x, y, label='predicted')

    axs[0].legend()

    axs[1].set_title("Error.")
    axs[1].grid(True)
    axs[1].plot(self.errors)

    plt.figsize=(10, 5)
    plt.show()
    
  def draw_network(self):
    G = nx.DiGraph()

    # Add neurons as nodes
    for neuron in self.neurons:
        G.add_node(neuron.name)

    # Add connections as edges
    for neuron1, neuron2 in self.connections:
        G.add_edge(neuron1.name, neuron2.name)

    # Create a position dictionary for nodes
    pos = {}

    # Assign positions for input layer nodes on the left
    input_layer_nodes = self.input_layer
    y_position = 0
    for neuron in input_layer_nodes:
        pos[neuron.name] = (0, y_position)
        y_position += 1

    # Assign positions for other neurons
    other_neurons = [neuron for neuron in self.neurons if neuron not in input_layer_nodes]
    num_other_neurons = len(other_neurons)
    y_position = num_other_neurons // 2
    for neuron in other_neurons:
        pos[neuron.name] = (1, y_position)
        y_position -= 1

    # Draw the network
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_size=1500, node_color="skyblue", font_size=12, arrows=True)
    plt.title("Neural Network Architecture")
    plt.show()
