"""

"mnist_nn.py"

Implementation of digit classifying neural networks: an MLP and a convolutional
network. All were written using numpy and scipy-- no traditional machine
learning languages were used (scikit-learn, TensorFlow, keras, PyTorch, etc.).

_______________________________________________________________________________

Accuracy (MLP): 98.20%
 - Structure: 784, 100, 10
 - Activation: sigmoid
 - Cost: cross-entropy, L2 regularization with lambda = 5.0
 - Learning rate: 0.1
 - Number of epochs: 60

Accuracy (convolutional network): []
 - Structure: Conv((5, 5), 20), Pooling((2, 2)), Dense(100), Dense(10)
 - Activation: relu, softmax in output layer
 - Cost: log-likelihood, L2 regularization with lambda = 0.1
 - Learning rate: 0.1
 - Number of epochs: 60

_______________________________________________________________________________

Progress for "mlp.py":

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

_______________________________________________________________________________

Progress for "conv_nn.py":

1. 6/18/19: created Layer, Conv, Pooling, and Dense classes, along with their
   respective feed-forward functions. As of now, they seem functional,
   but feeding forward through a convolutional network is taking around
   10 times longer than propagating through an MLP (for a single example).

2. 6/19/19: created Network class as implementation of Layer classes.
   Initially, propagating 10000 examples through a convolutional net
   with a convolutional layer (3 feature maps), a max-pooling layer,
   and a dense layer took 40 seconds (compared to ~0.7 seconds in a 3-layer
   MLP with 100 hidden neurons and the same input/output structure). This time
   was decreased to ~4.5 seconds by improving the max pooling function.
   However, propagating 60000 examples (~size of the MNIST dataset) took
   30 seconds, which is worrisome (compared to ~3.3 seconds in the MLP network).

3. 6/25/19: implemented backpropagation and SGD. However, training is unstable
   and inaccurate (accuracy with 1,000 MNIST images is 40%-70%), suggesting that
   there is an error with backpropagation/forward-propagation somewhere in
   the network. An MLP compiled with this program performs as well as an MLP
   created with "mlp.py", suggesting that there error lies in the
   convolutional and/or pooling layers.

   Structure: Layer((28, 28)), Conv((5, 5), 3), Pooling((2, 2)), Dense(10)
   Learning rate: 0.4
   Minibatch size: 20

4. 6/26/19: fixed backpropagation and pooling in convolutional layer. On 1,000
   train images, the network reaches ~81% accuracy after 5 epochs, slightly
   higher than the MLP's highest accuracy on the same training set (~80%).
   The convolutional network takes about 10 times longer to train but also
   yields greater accuracy in fewer epochs. Note that I am not interested in
   optimizing this network (i.e., trying to make it super-accurate)-- the
   building of this program was purely for educational purposes, not to
   produce a powerful network.

   Structure: Layer((28, 28)), Conv((5, 5), 20), Pooling((2, 2)), Dense(10)
   Learning rate: 0.4
   Minibatch size: 20

5. 6/28/19: network still broken. ~95% accuracy achieved on MNIST after 2 epochs
   (which took ~1.1 hours per epoch), which is too low, suggesting that there
   is something wrong with backpropagation. Additionally, small changes to
   weight initialization cause the network to become a noise machine, and
   online learning yields greater accuracy/stability than SGD.

   Structure: Layer((28, 28)), Conv((5, 5), 20), Pooling((2, 2)), Dense(100),
              Dense(10)
   Learning rate: 0.4
   Minibatch size: 10

6. 7/4/19: fixed forward propagation in max pooling layer but
   RuntimeWarnings galore (hooray...):
   
   Warning (from warnings module):
    File "/Users/ryan/Documents/Coding/neural-networks/src/mlp.py", line 79
      return 1.0 / (1.0 + np.exp(-z))
   RuntimeWarning: overflow encountered in exp

   Warning (from warnings module):
    File "/Users/ryan/Documents/Coding/neural-networks/src/mlp.py", line 57
      for (a, y) in pairs)) / len(pairs)
   RuntimeWarning: divide by zero encountered in log

   Warning (from warnings module):
    File "/Users/ryan/Documents/Coding/neural-networks/src/mlp.py", line 57
     for (a, y) in pairs)) / len(pairs)
   RuntimeWarning: invalid value encountered in multiply

   Strangely, these errors only occur when training on the full dataset.
   Unlike last time these errors occurred, the problem is not with the
   noramlization of the data.

7. 7/4/19: more efficient implementation of backpropagation and forward
   propagation using flattened weights and error vectors.

8. 7/5/19: decreased runtime significantly by using convolve2d instead of
   convolve. Comparison to previous assessments:
   - Propagating 10000 examples with the same (convolutional) structure as
     6/19/19 takes ~3.9 seconds compared to ~4.5 seconds as of 6/19/19
   - Propagating 60000 examples with the same structure as 6/19/19 takes ~19.5
     seconds compared to ~30 seconds as of 6/19/19

9. 7/6/19: fixed RuntimeWarnings from 7/4/19 by fixing calculation of error
   and nabla_b in the convolutional layer. This fix was a result of my noticing
   that the gradients were exploding in the convolutional layer. Comparison
   to previous assessments:
   - Training on 1,000 MNIST images for 5 epochs, a convolutional network with
     the same structure as 6/26/19 achieves ~82.5%-~84.5% accuracy. In contrast,
     an MLP network with the same structure as 6/26/19 achieves ~84%-~85%
     accuracy. An MLP network that implements relu instead of sigmoid in the
     hidden layer can achieve accuracy up to ~87.5%-~88.0% on the same training
     set. However, when run for 50 epochs, both the convolutional model and
     the MLP relu model stabilize at ~89% accuracy after around 30-40 epochs.

"""
#Libraries
import numpy as np
import mnist_loader

import sys
sys.path.insert(0, "/Users/ryan/Documents/Coding/neural-networks/src")
import mlp #vanilla feed-forward neural network
from mnist_conv_display import test #convolutional neural network + MLP network

#Testing area
if __name__ == "__main__":
  if input("Use \"mlp.py\" or \"conv_nn.py\"? (mlp/conv): ") == "mlp":
    data = mnist_loader.load_data("mlp")
    structure = [784, 100, 10]
    learning_rate = 0.1
    minibatch_size = 10
    num_epochs = 5
    momentum = None
    cost_function = mlp.Cost("cross-entropy", regularization = "L2",
                    reg_parameter = 5.0)
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
    net = test(net_type = input("MLP or ConvNN test? (mlp/conv): "),
               data = data, shorten = False, test_acc = True)
