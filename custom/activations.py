from scipy.special import expit
import numpy as np
  
class ActivationInterface:
  def __init__(self):
    self._name = None

  @property
  def name(self):
    assert self._name is not None
    return self._name
    
  def calc(self, x):
    raise NotImplementedError
    
  def deriv(self, x):
    raise NotImplementedError
    
class Sigmoid(ActivationInterface):  
  def __init__(self):
    self._name = "sigmoid"  
  
  @staticmethod
  def calc(x):
    return expit(x)
  
  def deriv(self, x):
    return self.calc(x)*(1-self.calc(x))


class Tanh(ActivationInterface):  
  def __init__(self):
    self._name = "tanh"  
  
  @staticmethod
  def calc(x):
    return np.tanh(x)

  @staticmethod
  def deriv(x):
    return 1 - np.tanh(x)**2

class ReLU(ActivationInterface):  
  def __init__(self):
    self._name = "relu"  
  
  @staticmethod
  def calc(x):
    return max(0, x)

  @staticmethod
  def deriv(x):
    return 1 if x > 0 else 0

class LeakyReLU(ActivationInterface):  
  def __init__(self, alpha=0.01):
    self._name = "leaky_relu"
    self._alpha = alpha
  
  @staticmethod
  def calc(x):
    return max(self._alpha*x, x)

  @staticmethod
  def deriv(x):
    return 1 if x > 0 else alpha
