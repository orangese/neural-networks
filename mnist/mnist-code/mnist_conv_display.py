
"""

"mnist_conv_display.py"

A program to display the results of a trained ConvNN.

"""

#Libraries
import sys
sys.path.insert(0, "/Users/ryan/Documents/Coding/neural-networks/src")
from conv_nn import Network, Layer, Conv, Pooling, Dense
sys.path.insert(0, "/Users/ryan/Documents/Coding/neural-networks/\
                 mnist/mnist-code")
from mnist_loader import load_data
import matplotlib.pyplot as plt
import numpy as np

#Testing area
def show_kernels(net):
  #displays learned weight kernels
  disp_layer = next((l for l in net.layers if isinstance(l, Conv)), None)
  disp_weights = np.copy(disp_layer.weights)
  fig, axes = plt.subplots(*closest_multiples(disp_weights.shape[0]))
  fig.canvas.set_window_title("Visualizing convolutional networks")
  fig.suptitle("Kernel weights for convolutional layer")
  for ax in axes.flatten():
    ax.imshow(disp_weights[list(axes.flatten()).index(ax)])
  plt.show()

def show_output(net, layer):
  disp_layer = next((l for l in net.layers if isinstance(l, layer)), None)
  disp_output = np.copy(disp_layer.output)
  fig, axes = plt.subplots(*closest_multiples(disp_layer.dim[0]))
  fig.canvas.set_window_title("Visualizing convolutional networks")
  fig.suptitle("Output for {0} layer".format(layer))
  for ax in axes.flatten():
    ax.imshow(disp_output[list(axes.flatten()).index(ax)])
  plt.show()

def closest_multiples(n):
  #returns two multiples of n that are closest together, n > 1
  factors = []
  for i in range(1, n):
    if n % i == 0: factors.append(((i, int(n / i)), (abs(i - int(n / i)))))
  index = list(zip(*factors))[1].index(min(list(zip(*factors))[1]))
  return factors[index][0]

if __name__ == "__main__":
  np.seterr(all = "raise")
  net = Network([Layer((28, 28)), Conv((5, 5), 20), Pooling((2, 2)), Dense(10)])
  data = load_data("conv")
  net.propagate(np.ones((28, 28)))
#  net.SGD(data["train"][:1000], 5, 0.1, 10, data["validation"][:1000])
  show_kernels(net)
  show_output(net, Conv)
  input()
