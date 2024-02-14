# Imports
from .neuron import Neuron
import matplotlib.pyplot as plt
import math
import random
import numpy as np
from scipy.special import expit
import networkx as nx
from copy import deepcopy

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
    self.exercices = kwargs.get("exercices", {})
    self.exercice_errors = []
    self.connections = []

    self._init_neurons(*kwargs["neurons"])
    self._init_inputs(*kwargs["inputs"])
    self._init_outputs(*kwargs["outputs"])
    self._init_connections(*kwargs["connections"])

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
      origin = self._neuron_dict[origin]
      destination = self._neuron_dict[destination]
      self.connect(origin, destination)

  def __str__(self):
    return f"network {self.name}"

  def __getitem__(self, key):
    return self._neuron_dict[key]

  def add_neuron(self, neuron):
    self.neurons.add(neuron)
    if neuron.name in self._neuron_dict:
      assert neuron is self._neuron_dict[neuron.name]
      return
    self._neuron_dict[neuron.name] = neuron

  def as_dict(self):
    neurons = []
    for neuron in self.neurons:
      neuron_params = {
            "name": neuron.name,
            "weights": [1],
            "bias": 0,
            "activation":neuron.activation,
            "derivative activation":neuron.d_activation,
        }
      neurons.append(neuron_params)

    inputs = []
    for neuron in self.input_layer:
      inputs.append(neuron.name)

    outputs = []
    for neuron in self.output_layer:
      outputs.append(neuron.name)

    connections = []
    for neuron1, neuron2 in self.connections:
      connections.append(
        (
          neuron1.name,
          neuron2.name,
          )
      )
    data = {
        "neurons": neurons,
        "inputs":inputs,
        "outputs":outputs,
        "connections":connections,
        "exercices": deepcopy(self.exercices),
    }
    return data

  def new(self, name):
    params = self.as_dict()
    params["name"] = name   
    copy = Network(**params)
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

  def train(self, repetitions, exercices=None):
    exercices = exercices or {} 
    exercices.update(self.exercices)
    inputs = exercices["inputs"]
    assert inputs
    targets = exercices["outputs"]
    tolerances = exercices["tolerance"]

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

    if self.exercices:
      x = [x[0] for x in self.exercices["inputs"]]
      y = [y[0] for y in self.exercices["outputs"]]
      axs[0].plot(x, y, label='target')

    x = [i/10 for i in range(-10, 250)]
    y = [self.predict([i]) for i in x]
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

    columns = {}
    for neuron in self.neurons:
      if neuron.position in columns:
        columns[neuron.position].append(neuron.name)
        continue
      columns[neuron.position] = [neuron.name]
    
    positions = {}
    for position, column in columns.items():
      for i, name in enumerate(column):
        positions[name] = (position, i)

    # Draw the network
    # plt.figure(figsize=(12, 8))
    nx.draw(G, positions, with_labels=True, node_size=1500, node_color="skyblue", font_size=12, arrows=True)
    plt.title("Neural Network Architecture")
    plt.show()
