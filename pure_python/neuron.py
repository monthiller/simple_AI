# Imports
import matplotlib.pyplot as plt
import math
import random
import numpy as np
from scipy.special import expit

dot = np.dot
_sigmoid  = expit

def _d_sigmoid( x):
  return _sigmoid(x)*(1-_sigmoid(x))

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

params = {
    "weights": [1],
    "bias": 0,
    "activation":_sigmoid,
    "derivative activation":_d_sigmoid,
}
neuron0 = Neuron(0, **params)
neuron0.plot()
###############################################################################
#Network class
class Network:
  def __init__(self, name, **kwargs):
    self.input_layer = list(kwargs["inputs"])
    self.output_layer = list(kwargs["outputs"])
    self.inputs = None
    self._target = None
    self.learning_rate = 0.1
    self.neurons = set(kwargs["neurons"])
    self.errors = list()
    self._neuron_dict = {}
    self.connections = []
    self.ready = False
    self.name = name
    self.predicted = None
    self.training = kwargs["training"]
    self.exercice_errors = []

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
    neuron2.previous_layer.append(neuron1)
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
    axs[0].set_title(f"Trained for {len(self.errors)} repetitions, with {network.learning_rate} learning rate.")
    axs[0].grid(True)

    x = [x[0] for x in self.training["inputs"]]
    y = [y[0] for y in self.training["outputs"]]
    axs[0].plot(x, y, label='target')

    x = [i/10 for i in range(-10, 250)]
    y = [network.predict([i]) for i in x]
    print(max(y))
    axs[0].plot(x, y, label='predicted')

    axs[0].legend()

    axs[1].set_title("Error.")
    axs[1].grid(True)
    axs[1].plot(self.errors)

    plt.figsize=(10, 5)
    plt.show()



#################################################################################
dataset = {
    "inputs":[],
    "outputs":[],
    "tolerance":[],
}
for i in range(50):
  input = 24*i/50
  if input<=12:
    dataset["inputs"].append([input,])
    dataset["outputs"].append([0,])
    dataset["tolerance"].append([0, 0.4])
    continue
  dataset["inputs"].append([input,])
  dataset["outputs"].append([1])
  dataset["tolerance"].append([0.6, 1])

training  = {
    "inputs":[],
    "outputs":[],
    "tolerance":[],
}
check  = {
    "inputs":[],
    "outputs":[],
    "tolerance":[],
}
for i, _ in enumerate(dataset["inputs"]):
  if i%2 == 0:
    check["inputs"].append(dataset["inputs"][i])
    check["outputs"].append(dataset["outputs"][i])
    check["tolerance"].append(dataset["tolerance"][i])
    continue
  training["inputs"].append(dataset["inputs"][i])
  training["outputs"].append(dataset["outputs"][i])
  training["tolerance"].append(dataset["tolerance"][i])

params = {
    "weights": [1],
    "bias": 0,
    "activation":_sigmoid,
    "derivative activation":_d_sigmoid,
}
neuron0 = Neuron(0, **params)
params = {
    "neurons": [
        neuron0
    ],
    "inputs":[neuron0],
    "outputs":[neuron0],
    "connections":[],
    "training":training,
}

network = Network(name=0, **params)
network.train(repetitions=1000)
network.test(check)
network.plot()
