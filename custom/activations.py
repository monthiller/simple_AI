from scipy.special import expit
  
class ActivationInterface:
    
  def calc(self, values):
    raise NotImplementedError
    
  def deriv(self, values):
    raise NotImplementedError
    
class Sigmoid(ActivationInterface):
  name = "sigmoid"
  
  @staticmethod
  def calc(values):
    return expit(values)

  
  def deriv(self, values):
    return self.calc(x)*(1-self.calc(x))
