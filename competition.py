class Competition:
  def __init__(self, **kwargs):
    self.deadline = kwargs["deadline"]
    self.coachs = []
    self._init_coachs(*kwargs["coachs"])
  
  def _init_coachs(self, *args):
    for params in args:
      coach = Coach(**params)
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
