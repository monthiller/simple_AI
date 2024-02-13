# Imports
import matplotlib.pyplot as plt
import math
import random
import numpy as np

dot = np.dot


# Neuron class
class Neuron:
  def __init__(self, name, **kwargs):
    self.name = name
    self.input_connections = []
    self.output_connections = []
    self.bias = kwargs["bias"]
    self.inputs = None
    self.weights = kwargs["weights"]
    self.correct = None
    self.predicted = None

    self.activation = kwargs["activation"]
    self.d_activation = kwargs["derivative activation"]

  def __repr__(self):
    return str(self.name)

  def copy(self, name):
    neuron = Neuron(name)
    neuron.weights = self.weights.copy()
    neuron.bias = self.bias
    return neuron

  def find_paths(self, neuron):
    paths = set()

    for i, previous in enumerate(self.input_connections):
      if neuron is previous:
        paths.add(i)
        continue
      if neuron in previous.neurons:
        paths.add(i)

    return paths

  @property
  def neurons(self):
    neurons = set()
    neurons.update(self.input_connections)
    for neuron in self.input_connections:
        neurons.update(neuron.input_connections)
    return neurons

  def dp_w(self, neuron, i,):

    a = self.linear()

    b = self.d_activation(a)

    c = self.da_w(neuron,i)

    return b*c

  def da_w(self, neuron, i):
    if not self.input_connections:
      return self.inputs[0]

    paths = self.find_paths(neuron)
    if neuron is self:
      assert not paths, f"{paths=}"

      return self.input_connections[i].predict()

    # raise NotImplementedError()
    previous_errors = []
    for j in paths:
      previous_neuron = self.input_connections[j]
      previous_errors.append(self._w[j] * previous_neuron.dp_w(neuron, i))
    return sum(previous_errors)

  def dp_b(self, neuron):
      a = self.linear()
      b = self.d_activation(a)
      c = self.da_b(neuron)
      return b*c

  def da_b(self, neuron):
    if not self.input_connections:
      return 1
      # raise NotImplementedError(f"{neuron=}")

    paths = self.find_paths(neuron)

    if neuron is self:
      assert not paths, f"{paths=}"
      return 1

    previous_errors = []
    for j in paths:
      previous_neuron = self.input_connections[j]
      previous_errors.append(self._w[j] * previous_neuron.dp_b(neuron))

    return sum(previous_errors)

  def linear(self):
    x = self.inputs
    w = self.weights
    b = self.bias
    assert len(w) == len(x), f"{self=}, {w=}, {x=}"
    return  dot(w, x) + b

  def predict(self, inputs=None):
    if self.predicted is not None:
      return self.predicted

    if inputs is not None:
      self.inputs = inputs
    else:
      if self.inputs is None:
        assert self.previous_layer
        self.inputs = []
        for neuron in self.previous_layer:
          self.inputs.append(neuron.predict())

    value = self.linear()
    return self.activation(value)

  def plot(self):
    x = list(i/10 for i in range(-100, 300))
    y = []
    for i in x:
      y.append(self.predict(inputs=[i]))

    plt.plot(x, y)
    plt.title(f"Neuron {self.name}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()
