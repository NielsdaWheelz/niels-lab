import torch
from micrograd import Value
import numpy as np

class Neuron:
  def __init__(self, neuron_inputs):
    self.w = [Value(np.random.uniform(-1,1)) for _ in range(neuron_inputs)]
    self.b = Value(np.random.uniform(-1,1))

  def __call__(self, x):
    activation = sum(((wi * xi) for wi, xi in zip(self.w, x)), self.b)
    out = activation.tanh()
    return out

class Layer:
  def __init__(self, neuron_inputs, neuron_outputs): # neuron_inputs is the number of inputs to the layer, neuron_outputs is the number of neurons in the layer
    self.neurons = [Neuron(neuron_inputs) for _ in range(neuron_outputs)]
  
  def __call__(self, x):
    outs = [n(x) for n in self.neurons]
    return outs[0] if len(outs) == 1 else outs

class MLP:
  def __init__(self, nin, nouts): # nin is the number of inputs to the MLP, nouts is a list of the number of neurons in EACH layer
    size = [nin] + nouts # size is a list of the number of neurons in each layer
    self.layers = [Layer(size[i], size[i+1]) for i in range(len(nouts))] # create a list of layers, each layer is a list of neurons, iterate over the pairs of adjacent neurons in the size list

  def __call__(self, x):
    for layer in self.layers: # iterate over the layers
      x = layer(x)
    return x # return the output of the MLP

x = [2.0, 3.0, -1.0]
n = MLP(3, [4, 4, 1])
o = n(x)
print(o.data)