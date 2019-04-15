
"""

"mnist_test.py"

Program to test mnist_loader2.py and mnist_loader.py

"""

import mnist_loader2
import mnist_loader
import numpy as np
from timeit import default_timer as timer #for timing stuff
import digit_classifier_nn as net

network = net.Network([784, 30, 10], cost_function = net.Cost("log-likelihood",
                                                        regularization = "L2",
                                                        reg_parameter = 5.0),
                             body_activation = net.Activation("sigmoid"),
                             output_activation = net.Activation("softmax"))

start = timer()
data1 = mnist_loader.load_data()
end = timer()

print (end - start)

start = timer()
data2 = mnist_loader2.load_data()
end = timer()

print (end - start)

start = timer()
for i in range(len(data1["train"])):
  network.feed_forward(data1["train"][i][0])
for i in range(len(data1["validation"])):
  network.feed_forward(data1["validation"][i][0])
for i in range(len(data1["test"])):
  network.feed_forward(data1["test"][i][0])
end = timer()
print (end - start)

start = timer()
for i in range(len(data2["train"])):
  network.feed_forward(data2["train"][i][0])
for i in range(len(data2["validation"])):
  network.feed_forward(data2["validation"][i][0])
for i in range(len(data2["test"])):
  network.feed_forward(data2["test"][i][0])
end = timer()
print (end - start)

for i in range(10):
  print (np.argmax(data1["train"][i][1]), np.argmax(data2["train"][i][1]))
