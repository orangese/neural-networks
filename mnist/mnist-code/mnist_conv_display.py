
"""

"mnist_conv_display.py"

A program to display the results of a trained ConvNN.

"""

#Libraries
import sys
sys.path.insert(0, "/Users/ryan/Documents/Coding/neural-networks/src")
from conv_nn import Network, Layer, Conv, Pooling, Dense, test
sys.path.insert(0, "/Users/ryan/Documents/Coding/neural-networks/\
                 mnist/mnist-code")
from mnist_loader import load_data, display_image
import matplotlib.pyplot as plt
import numpy as np

#Testing area
def display(net, show_kernel = True, layer = Conv):
  """displays output of a layer or kernel weights using plt.imshow"""
  if show_kernel: assert layer is Conv, "can only show kernel for Conv layer"
  disp_layer = next((l for l in net.layers if isinstance(l, layer)), None)
  disp_obj = np.copy(disp_layer.weights if show_kernel else disp_layer.output)
##  num_fmaps = None if layer is Layer else disp_layer.dim[0]
##  fig, axes = plt.subplots(*closest_multiples(disp_item.shape[0] if show_kernel
##                                              else num_fmaps))
  fig, axes = plt.subplots() if layer is Layer \
              else plt.subplots(*closest_multiples(disp_layer.dim[0]))
  fig.canvas.set_window_title("Visualizing convolutional networks")
  if show_kernel: fig.suptitle("Kernel weights for {0} layer".format(layer))
  else: fig.suptitle("Output for {0} layer".format(layer))
  try:
    for ax in axes.flatten():
      ax.imshow(disp_obj[list(axes.flatten()).index(ax)], cmap = "gray")
  except AttributeError:
    axes.imshow(disp_obj, cmap = "gray")
  plt.show() 

def closest_multiples(n):
  """returns two multiples of n that are closest together"""
  if n == 1: return () #engineered to work with plt.subplots: () represents 1 plot
  factors = []
  for i in range(1, n):
    if n % i == 0: factors.append(((i, int(n / i)), (abs(i - int(n / i)))))
  return factors[np.argmin(list(zip(*factors))[1])][0]

if __name__ == "__main__":
  data = load_data("conv")
  net = test(net_type = input("MLP or ConvNN test? (mlp/conv): "), data = data)
  display(net)
  display(net, show_kernel = False, layer = Layer)
  display(net, show_kernel = False, layer = Conv)
  display(net, show_kernel = False, layer = Pooling)
  input()
