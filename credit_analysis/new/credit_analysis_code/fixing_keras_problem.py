"""

This code fixes the problem in keras where model.predict did not work
when rendering/posting. See instructions


"""

import tensorflow as tf
import keras
import numpy as np

#copy this function into "models.py"
def predict(parsed_model, vars_):
  return feed_forward(parsed_model, np.array([float(var) for var in vars_]))

#also goes in "models.py"
def parse_net(model):
  weights = np.array([model.layers[i].get_weights()[0]
                      for i in range(len(model.layers))])
  weights = weights.reshape(weights.shape[0], weights.shape[2],
                            weights.shape[1])
  biases = np.array([model.layers[i].get_weights()[1]
                     for i in range(len(model.layers))])
  biases = biases.reshape(biases.shape[0], biases.shape[1], 1)
  print (biases.shape, weights.shape)
  return (biases, weights)

#also goes in "models.py"
def feed_forward(parsed_model, a):
  a = a.reshape(len(a), 1)
  biases = parsed_model[0]
  weights = parsed_model[1]

  sigmoid = lambda z : 1.0 / (1.0 + np.exp(z))
  for b, w in zip(biases, weights):
    a = sigmoid(np.dot(w, a) + b)
  return a

#implementation instructions:

#step one: creating the sample model
#("model" should be replaced with the credit analysis network in "models.py")
model = keras.Sequential([
  keras.layers.Dense(10, input_shape = (6, ), activation = "sigmoid")
])
model.compile(loss = "binary_crossentropy", optimizer = "adam",
              metrics = ["accuracy"])

#step two: fake data
#replace with real independent variables that will be fed into net
fake_independent_vars = np.array([1, 2, 3, 4, 5, 6])

#step three: parse model in "models.py"
parsed_model = parse_net(model)
#don't forget, this line (and everything above it) goes in "models.py"!

#step four: go to "view.py" and import feed_forward and import parsed_model
print (feed_forward(parsed_model, fake_independent_vars))
