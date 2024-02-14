from scipy.special import expit
  
class ActivationInterface:
  @property
  @classmethod
  def name(cls):
    assert cls._name is not None
    return cls._name
    
  def calc(self, values):
    raise NotImplementedError
  def deriv(self, values):
    raise NotImplementedError
    
class Sigmoid(ActivationInterface):
  _name = "sigmoid"
  
  @staticmethod
  def calc(values):
    return expit(values)

  @staticmethod
  def deriv( values):
    return self.calc(x)*(1-self.calc(x))
