"""

This code fixes the problem in Keras where model.predict did not work
when rendering/posting. See instructions for more details.


"""

import keras
import numpy as np

#copy this function into "models.py"
def predict(parsed_model, vars_):
  #returns a, b, c, d, e, f, or g based on network prediction
  range_ = ["A", "B", "C", "D", "E", "F", "G"]
  prediction = feed_forward(parsed_model,
                            np.array([float(var) for var in vars_]))
  return range_[np.argmax(prediction)]

#also goes in "models.py"
def parse_net(model):
  #returns a model's weights and biases
  weights = np.array([model.layers[i].get_weights()[0]
                      for i in range(len(model.layers))])
  weights = weights.reshape(weights.shape[0], weights.shape[2],
                            weights.shape[1])
  biases = np.array([model.layers[i].get_weights()[1]
                     for i in range(len(model.layers))])
  biases = biases.reshape(biases.shape[0], biases.shape[1], 1)
  return (biases, weights)

#also goes in "models.py"
def feed_forward(parsed_model, a):
  #feed-forwards the input, "a", through a skeleton network
  #which will only have weights and biases (not a Keras object)
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
#this step goes in "models.py"
model = keras.Sequential([
  keras.layers.Dense(7, input_shape = (5, ), activation = "sigmoid")
])

#step two: fake data
#replace with real independent variables that will be fed into net
#this step goes in "view.py"
fake_independent_vars = np.array([1, 2, 3, 4, 5])

#step three: parse model in "models.py"
#this step goes in "models.py"
parsed_model = parse_net(model)

#step four: go to "view.py" and import the "predict" function
#and import the "parsed_model" variable
#this step goes in "view.py"
print (predict(parsed_model, fake_independent_vars))
