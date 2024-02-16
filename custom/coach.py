import matplotlib.pyplot as plt
from .network import Network
import random

class Coach:
  def __init__(self,name, **kwargs):
    self.name = name
    self.seed = kwargs["seed"]
    self.profile = Network(name="profile", **kwargs["profile"])
    self.dataset = kwargs["dataset"]
    self.number_of_selections = kwargs["number_of_selections"]
    self.sessions_size = kwargs["sessions_size"]
    self.canditate_number = kwargs["canditate_number"]
    self.team = []
    # self.selected = None
    self.first = None
    self.exercices = {}
    self.check = {}
    self.split_dataset(**kwargs["split_dataset"])
    self.sessions_to_tolerance = None

  @property
  def min_error(self):
    min_errors = []
    for network in self.team:
      if not network.errors:
        continue
      min_errors.append(min(network.errors))
      
    if not min_errors:
      return 10000000
    return min(min_errors)
  
  def split_dataset(self, pick_1_out_of, ):      
    self.exercices.clear()
    self.check.clear()

    for key in self.dataset:
      self.exercices[key] = []
      self.check[key] = []

    for i, line in enumerate(self.dataset["inputs"]):
      if i%pick_1_out_of == 0:
        for key in self.dataset:
          self.check[key].append(self.dataset[key][i])
        continue
      for key in self.dataset:
          self.exercices[key].append(self.dataset[key][i])

  def is_good_candidate(self, network, length):
    repetition_limits = list(i*length/10 for i in range(10))
    error_limits = list(i*10 for i in range(10, 0, -1))
    learning_rates = list(i/10 for i in range(10, 0, -1))

    for repetition_limit, error_limit, lr in zip(repetition_limits, error_limits, learning_rates, strict=True):
      if len(network.errors)<=repetition_limit:
        if min(network.errors) < error_limit:
          network.learning_rate = lr
          return True
        return False

  def build_network(self, name):
    network = self.profile.new(name=name)

    network.learning_rate = 1
    network.random_fit()
    return network

  def build_team(self):
    canditate_number = self.canditate_number
    self.team = [x for x in self.team if self.is_good_candidate(x, canditate_number)]
    possible_names = set(range(canditate_number))
    existing_names = set(network.name for network in self.team)
    available_name = possible_names.difference(existing_names)
    # size = total_candidates - len(self.team)
    for i in available_name:
      network = self.build_network(i)
      self.team.append(network)
  
  def train(self):
    for _ in self.train_and_report():
      pass
    comment = """
    random.seed(self.seed)
    repetitions = self.sessions_size
    number_of_selections = self.number_of_selections
    for i in range(number_of_selections):
      self.build_team()
      failed = set()
      for network in self.team:
        try:
          network.train(
              exercices=self.exercices,
              repetitions=repetitions)
          network.test(
              self.check
          )
        except OverflowError:
          failed.add(network)
          pass

        if network.ready:
          self.first = network
          self.team.remove(network)
          break

      assert len(failed)<len(self.team)
      for network in failed:
        self.team.remove(network)

      # self.team.sort(key=lambda network: network.errors[-1])

      if self.first:
        print(f"A solution was found after {i+1} selections!")
        self.team = [x for x in self.team if x.errors]
        self.team.sort(key=lambda network: network.errors[-1])
        self.sessions_to_tolerance = i
        break
      """

  def train_and_report(self):
    random.seed(self.seed)
    repetitions = 10
    number_of_selections = self.number_of_selections
    for i in range(number_of_selections):
      self.build_team()
      failed = set()
      for network in self.team:
        try:
          network.train(
              exercices=self.exercices,
              repetitions=repetitions)
          network.test(
              self.check
          )
        except OverflowError:
          failed.add(network)
          pass

        if network.ready:
          self.first = network
          self.team.remove(network)
          break
        yield

      assert len(failed)<len(self.team)
      for network in failed:
        self.team.remove(network)

      # self.team.sort(key=lambda network: network.errors[-1])

      if self.first:
        print(f"A solution was found after {i+1} selections!")
        self.team = [x for x in self.team if x.errors]
        self.team.sort(key=lambda network: network.errors[-1])
        self.sessions_to_tolerance = i
        break
  
  def get_podium(self):
    if self.first:
      return [self.first] + self.team[:2]
    return self.team[:3]

  def plot(self):
    print(f"Report for {self.name}.")
    if self.first:
      print(f"Solution was trained for {len(self.first.errors)} exercice sessions of {self.sessions_size} repetitions.")
      print(f"Solution has a learning rate of {self.first.learning_rate}.")
    else:
      print("Not solutions was found!")
    
    podium = self.get_podium()
    for place, network in enumerate(podium):
        plt.plot(network.errors, label=f'{place+1}, lr={network.learning_rate}')
    plt.title("Error.")
    plt.grid(True)
    plt.legend()
    plt.show()
