
"""

This a program to plot the progress and results of the digit classifying
neural network.

"""

#Libraries
import mnist_loader2 as mnist #for loading MNIST database
import digit_classifier_nn as net #for getting the results of the digit classifier
import matplotlib.pyplot as plt #for displaying results
import numpy as np #for arrays

data = mnist.load_data()

structure = [784, 100, 10]
cost_function = net.Cost("cross-entropy", regularization = "L2",
                         reg_parameter = 5.0)
digit_classifier = net.Network(structure,
                               cost_function = cost_function)

num_epochs = 60
learning_rate = 0.31
minibatch_size = 10
monitor = True
early_stopping = "aGL"
stop_parameter = 0.0
aGL_parameter = 100

print ("Evaluation without training: {0}%".format(
    digit_classifier.evaluate_accuracy(data["test"])))
  
print ("""Learning rate: {0}\nMinibatch size: {1}\
      \nNumber of epochs: {2}\nStructure: {3}\nCost function: {4}\nBody activation function: {5}\nOutput activation function: {6}"""
       .format(learning_rate, minibatch_size, num_epochs,
        digit_classifier.layers, digit_classifier.cost.name,
        digit_classifier.activation.name,
        digit_classifier.output_activation.name))
print ("Training in process...")

evaluation = digit_classifier.SGD(data["train"], num_epochs,
                                  learning_rate, minibatch_size,
                                  validation_data = data["validation"],
                                  test_data = data["test"],
                                  monitor = monitor,
                                  early_stopping = early_stopping,
                                  stop_parameter = stop_parameter,
                                  aGL_parameter = aGL_parameter)

actual = len(evaluation["validation accuracy"])
offset = 0
figure, plots = plt.subplots(1, 2)
figure.canvas.set_window_title("Performance of digit classifier")

plots[0].plot(np.asarray([i + 1 + offset for i in range(actual - offset)]).
           reshape(actual - offset, 1), evaluation["validation accuracy"]
           [offset:], color = "blue", label = "Validation accuracy")
plots[0].plot(np.asarray([i + 1 + offset for i in range(actual - offset)]).
           reshape(actual - offset, 1), evaluation["train accuracy"][offset:],
           color = "black", label = "Train accuracy")
plots[0].set_xlabel("Number of epochs")
plots[0].set_ylabel("Accuracy (%)")

plots[0].grid(True, linestyle = "--")
plots[0].legend(loc = "best")
plots[0].set_title("Accuracy vs. epochs")

plots[1].plot(np.asarray([i + 1 + offset for i in range(actual - offset)]).
              reshape(actual - offset, 1), evaluation["validation cost"][offset:],
              color = "blue", label = "Validation cost")
plots[1].plot(np.asarray([i + 1 + offset for i in range(actual - offset)]).
              reshape(actual - offset, 1), evaluation["train cost"][offset:],
              color = "black", label = "Train cost")
plots[1].set_xlabel("Number of epochs")
plots[1].set_ylabel("Cost (cross-entropy)")

plots[1].grid(True, linestyle = "--")
plots[1].legend(loc = "best")
plots[1].set_title("Cost vs. epochs")

plt.show()
