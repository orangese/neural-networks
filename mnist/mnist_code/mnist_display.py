
"""

"mnist_display.py"

A program to plot the progress and results of the digit classifying
neural network.

"""

#Libraries
import src.mlp as net #for getting the results of the digit classifier
import matplotlib.pyplot as plt #for displaying results
import numpy as np #for arrays

#Main
structure = [784, 100, 10]
cost_function = net.Cost("log-likelihood", regularization = "L2", reg_parameter = 5.0)
body_activation = net.Activation("sigmoid")
output_activation = net.Activation("softmax")
num_epochs = 30
learning_rate = 0.1
minibatch_size = 10
large_weight_initialization = False
monitor = True
early_stopping = ["aGL", 0.0, 50]
lr_variation = ["average_GL", 0.0, 10, 2, 0.002]
write = False
#use the above to change SGD stuff-- not in the main methods!

model = net.Network(structure, cost_function = cost_function, body_activation = body_activation,
                    output_activation = output_activation, weight_init = large_weight_initialization)
evaluation = model.train(structure, learning_rate, minibatch_size, num_epochs, monitor = True, write = write,
                         early_stopping = early_stopping, lr_variation = lr_variation)

def accuracy_and_cost(evaluation, offset):
  actual = len(evaluation["validation accuracy"])
  figure, plots = plt.subplots(1, 2)
  figure.canvas.set_window_title("Performance of digit classifier")

  plots[0].plot(np.asarray([i + 1 + offset for i in range(actual - offset)]).reshape(actual - offset, 1),
                evaluation["validation accuracy"][offset:], color = "blue", label = "Validation accuracy")
  plots[0].plot(np.asarray([i + 1 + offset for i in range(actual - offset)]).reshape(actual - offset, 1),
                evaluation["train accuracy"][offset:], color = "black", label = "Train accuracy")
  plots[0].set_xlabel("Number of epochs")
  plots[0].set_ylabel("Accuracy (%)")

  plots[0].grid(True, linestyle = "--")
  plots[0].legend(loc = "best")
  plots[0].set_title("Accuracy vs. epochs")

  plots[1].plot(np.asarray([i + 1 + offset for i in range(actual - offset)]).reshape(actual - offset, 1),
                evaluation["validation cost"][offset:], color = "blue", label = "Validation cost")
  plots[1].plot(np.asarray([i + 1 + offset for i in range(actual - offset)]).reshape(actual - offset, 1),
                evaluation["train cost"][offset:], color = "black", label = "Train cost")
  plots[1].set_xlabel("Number of epochs")
  plots[1].set_ylabel("Cost")

  plots[1].grid(True, linestyle = "--")
  plots[1].legend(loc = "best")
  plots[1].set_title("Cost vs. epochs")

  plt.show()

def weight_initialization(model, evaluation):
  print ("\nWith large weight initialization:")
  large_evaluation = model.train(structure, learning_rate, minibatch_size, num_epochs, monitor = True, write = write,
                                 early_stopping = early_stopping, lr_variation = lr_variation)

  actual = len(evaluation["validation accuracy"])
  figure = plt.gcf()
  figure.canvas.set_window_title("Performance of digit classifier")

  plt.plot(np.asarray([i + 1 for i in range(actual)]).reshape(actual, 1), evaluation["validation accuracy"],
           color = "blue", label = "Accuracy with regular weight initializer")
  plt.plot(np.asarray([i + 1 for i in range(actual)]).reshape(actual, 1), large_evaluation["validation accuracy"],
           color = "black", label = "Accuracy with large weight initializer")
  plt.xlabel("Number of epochs")
  plt.ylabel("Accuracy (%)")

  plt.grid(True, linestyle = "--")
  plt.legend(loc = "best")
  plt.title("Accuracy vs. epochs")

  plt.show()

if __name__ == "__main__":
  weight_initialization(evaluation)
