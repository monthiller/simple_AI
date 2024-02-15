import custom.coach
from time import time

class Competition:
  coach_classes = {
    "custom": custom.coach.Coach
  }
  def __init__(self, **kwargs):
    self.deadline = kwargs["deadline"]
    self.coachs = []
    self._init_coachs(*kwargs["coachs"])
  
  def _init_coachs(self, *args):
    for params in args:
      coach_class = params["coach_class"]
      coach_class = Competition.coach_classes[coach_class]
      coach = coach_class(**params)
      self.coachs.append(coach)

  def compete(self):
    deadline = self.deadline
    start = time()    
    iterators = [
        coach.train_and_report()
        for coach
        in self.coachs
        ]
    
    while True:
      if deadline < time() - start:
        break
      try:
        for iterator in iterators:
          next(iterator)
      except StopIteration:
        break
    
    for coach in self.coachs:
      coach.plot()
