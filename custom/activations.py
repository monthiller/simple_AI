from scipy.special import expit
  
class ActivationInterface:
  @property
  def name(self):
    assert self._name is not None
    return self._name
    
  def calc(self, values):
    raise NotImplementedError
  def deriv(self, values):
    raise NotImplementedError
    
class Sigmoid(ActivationInterface):
  def __init__(self):
    self._name = "sigmoid"
  
  @staticmethod
  def calc(values):
    return expit(values)

  @staticmethod
  def deriv( values):
    return self.calc(x)*(1-self.calc(x))
