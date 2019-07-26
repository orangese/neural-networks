"""

"mnist_nn.py"

Implementation of digit classifying neural networks: an MLP and a convolutional network. All were written using numpy
and scipy-- no traditional machine learning languages were used (scikit-learn, TensorFlow, keras, PyTorch, etc.).

______________________________________________________________________________________________________________________

Accuracy (MLP): 98.20%
 - Structure: 784, 100, 10
 - Activation: sigmoid
 - Cost: cross-entropy, L2 regularization with lambda = 5.0
 - Learning rate: 0.1
 - Number of epochs: 60

Accuracy (CNN): []
 - Structure: Layer((28, 28)), Conv((5, 5), 20), Pooling((2, 2)), Dense(100), Dense(10)
 - Activation: relu, softmax in output layer
 - Cost: log-likelihood, L2 regularization with lambda = 0.1
 - Learning rate: 0.1
 - Number of epochs: 60

"""

# Libraries
from mnist.mnist_code.mnist_loader import load_data

import sys
sys.path.insert(0, "/Users/ryan/Documents/Coding/neural-networks/src")
import src.mlp as mlp # vanilla feed-forward neural network
from mnist.mnist_code.mnist_conv_display import test # convolutional neural network + MLP network

# Testing area
if __name__ == "__main__":
  if input("Use \"mlp.py\" or \"conv_nn.py\"? (mlp/conv): ") == "mlp":
    data = load_data("mlp")
    structure = [784, 100, 10]
    learning_rate = 0.1
    minibatch_size = 10
    num_epochs = 5
    momentum = None
    cost_function = mlp.Cost("cross-entropy", regularization = "L2", reg_parameter = 5.0)
    output_activation = mlp.Activation("sigmoid")
    weight_init = "regular"
    write = None
    lr_variation = None
    early_stopping = None
    dropout = None

    classifier = mlp.Network(structure, cost_function = cost_function, output_activation = output_activation,
                             weight_init = weight_init)
    classifier.train(data, learning_rate, minibatch_size, num_epochs, momentum = momentum, dropout = dropout,
                     early_stopping = early_stopping, lr_variation = lr_variation, monitor = False, show = True,
                     write = write)
  else:
    data = load_data("conv")
    net = test(net_type = input("MLP or ConvNN test? (mlp/conv): "), data = data, shorten = False, test_acc = True)
