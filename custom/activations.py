from scipy.special import expit
  
class ActivationInterface:
  def calc(self, values):
    raise NotImplementedError
  def deriv(self, values):
    raise NotImplementedError
    
class Sigmoid(ActivationInterface):
  @staticmethod
  def calc(values):
    return expit(values)

  @staticmethod
  def deriv( values):
    return self.calc(x)*(1-self.calc(x))
