"""

"credit_analysis_nn.py"

A program that uses keras to build and train a neural network to grade
the quality of a loan.

Accuracy: 94.95% (25 epochs)

"""

#Libraries
from credit_analysis_loader import load_data
import keras

#Neural network (using keras)
def build():
  reg = keras.regularizers.l2(0.001)
  model = keras.Sequential([
    keras.layers.Dense(20, input_shape = (24, ), activation = "sigmoid",
                       kernel_regularizer = None),
    keras.layers.Dense(20, input_shape = (20, ), activation = "sigmoid",
                       kernel_regularizer = None),
    keras.layers.Dense(7, input_shape = (20, ), activation = "softmax",
                      kernel_regularizer = None)])
  model.compile(loss = "binary_crossentropy", optimizer = "adam",
                metrics = ["accuracy"])
  
  return model

def train(data, model):
  tr_X, tr_Y = data["train"]
  
  model.fit(tr_X, tr_Y,
            epochs = 25, batch_size = 32, verbose = 2,
            validation_data = data["validation"])
  
  return model

def evaluate(data, model):
  te_X, te_Y = data["test"]

  evaluation = model.evaluate(te_X, te_Y, verbose = 0)
  print ("Accuracy: {0}%".format(round(evaluation[1] * 100, 2)))

  return evaluation

def save(model, filename):
  model.save(filename)

def load_model(filename):
  model = keras.models.load_model(filename)

#Testing area
if __name__ == "__main__":
  data, cols, big_data = load_data(0.8, keras_ = True)

  print ("Starting training")
  model = train(data, build())
  evaluation = evaluate(data, model)
