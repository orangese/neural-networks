"""

"mnist_nn.py"

A digit classifying neural network.

Accuracy: 98.20% (60 epochs)

Progress:
1. 2/10/19: program running but encountering error:

  "Warning (from warnings module):
  File "/Users/ryan/Library/Mobile Documents/com~apple~CloudDocs/Coding/
  digit_classifier.py", line 134
    return 1.0/(1.0 + np.exp(-z)) #np.exp(z) returns e^z
  RuntimeWarning: overflow encountered in exp"

  (maybe as a result of the error), the performance of the network is abysmal
  (10.32% after 25 epochs).
  
  Structure: 784, 30, 10
  Learning rate: 3.0
  Minibatch size: 30

2. 2/12/19: program running but encountering same error as 2/10/19.
  The performance has increased slightly after a fixing of the backpropagation
  algorithm-- the indexing on the nabla_w variable did not match that of the
  self.weights variable. 40.40% after 25 epochs, peak of 47.45% at epoch 20.
  The accuracy fluctuates widly; the only real progress seems to be between
  epochs 0 and 1 (+~30%). Accuracy at epoch 1 was 43.39%; the final accuracy
  remains mostly unchanged.
  
  Structure: 784, 30, 10
  Learning rate: 3.0
  Minibatch size: 30

3. 2/13/19: program running but encountering same error as 2/10/19.
  Performance has increased significantly after an adjustment of the learning
  rate from 3.0 to 0.25. 83.57% after 25 epochs, peak of 92.73% after 249 epochs.
  However, accuracy is still not optimal; performance should reach around 90%
  after 1 epoch (first epoch is ~30%) and should reach its peak (~95%) at around
  25 epochs.
  
  Structure: 784, 30, 10
  Learning rate: 0.25
  Minibatch size: 30

4. 2/18/19: same error; performance increased to 93.70% after 446 epochs. 84.41%
  after 25 epochs.

  Structure: 784, 50, 10
  Learning rate: 0.2
  Minibatch size: 25

5. 3/23/19: error was discovered to be with the data loading program. As a result,
  expected results were achieved (i.e., ~95% with MSE) and the overflow error was
  eliminated. Cross-entropy loss, log-likelihood loss (plus a softmax output
  layer), early stopping, and L1 and L2 regularization were implemented.
  Performance increased to 98.02% after 49 epochs. 97.76% after 25 epochs.

  Structure: 784, 100, 10
  Cost function: cross-entropy loss
  Learning rate: 0.3
  Minibatch size: 10
  Regularization parameter: 0.5
  Early stopping: aGL
  Early stopping parameter(s): 100, 0.0

6. 4/15/19: rewrite of the mnist_loader.py program to fix above error
  (previously, the mnist_loader used was from the book, "Neural Networks and
  Deep Learning"). The key change was the addition of a normalizer function,
  which converted the value of the pixels (0-255) to a sigmoidal range (0-1).
  The reason for the previous np.exp() overflow error was the failure to
  normalize.

"""
#Libraries
import numpy as np
import mnist_loader

import sys
sys.path.insert(0, "/Users/ryan/Documents/Coding/neural-networks/src")
import mlp #vanilla feed-forward neural network
import conv_nn #convolutional neural network as well as a MLP network

#Testing area
if __name__ == "__main__":
  if input("Use \"mlp.py\" or \"conv_nn.py\"? (mlp/conv): ") == "mlp":
    data = mnist_loader.load_data("mlp")
    data["train"] = data["train"][:1000]
    structure = [784, 100, 10]
    learning_rate = 0.1
    minibatch_size = 20
    num_epochs = 5
    momentum = None
    cost_function = mlp.Cost("cross-entropy", regularization = "L2",
                    reg_parameter = 2.0)
    output_activation = mlp.Activation("sigmoid")
    weight_init = "regular"
    write = None
    lr_variation = None
    early_stopping = None
    dropout = None

    classifier = mlp.Network(structure, cost_function = cost_function,
                             output_activation = output_activation,
                             weight_init = weight_init)
    classifier.train(data, learning_rate, minibatch_size, num_epochs,
                     momentum = momentum, dropout = dropout,
                     early_stopping = early_stopping,
                     lr_variation = lr_variation, monitor = False,
                     show = True, write = write)
  else:
    data = mnist_loader.load_data("conv")
##    data["train"] = data["train"][:1000]
##    data["test"] = data["test"][:1000]
##    data["validation"] = data["validation"][:1000]
    conv_nn.test(data = data, test_acc = True,
                 net_type = input("MLP or ConvNN test? (mlp/conv): "))
