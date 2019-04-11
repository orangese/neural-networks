"""

"digit_classifier.py" is a digit classifying neural network.

Type: vanilla neural network (MLP feedforward)
Activation function(s): sigmoid, softmax
Architecture: chosen by user
Cost function(s): MSE, cross-entropy, log-likelihood
Training: SGD and vanilla BP
Early stopping: GL, aGL
Regularization: 
Accuracy: 98.02%
Hyperparameters: chosen by user

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
    
"""

#Libraries
import mnist #for loading the MNIST data (not a standard library)
import numpy as np #for fast matrix-based computations

#Classes
class Cost(object):
  #class for the different cost functions

  def __init__(self, name, regularization = None, reg_parameter = None):
    self.name = name
    self.regularization = regularization
    self.reg_parameter = reg_parameter

  def derivative(self, a, y):
    #returns ∂Cx/∂a_L
    if self.name == "mse":
      return (a - y)
    elif self.name == "cross-entropy":
      return ((a - y) / (a * (1 - a)))
    elif self.name == "log-likelihood":
      return (y / a)

  def calculate(self, pairs):
    #accepts a list of tuples (a, y) and returns average cost over that list
    if self.name == "mse":
      return sum(np.linalg.norm(a - y) ** 2.0 for (a, y) in pairs) \
             / (2.0 * len(pairs))
    elif self.name == "cross-entropy":
      return sum(sum(np.nan_to_num(-y * np.log(a) - (1.0 - y) * np.log(1.0 - a)
                 for (a, y) in pairs))) / (len(pairs))
    elif self.name == "log-likelihood":
      return sum(np.log(np.argmax(a)) for (a, y) in pairs) \
             / (-1.0 * len(pairs))

  def get_error(self, activation, activations, weighted_inputs, label):
    if self.name == "mse":
      return self.derivative * activation.calculate(weighted_inputs)
    elif self.name == "cross-entropy" or self.name == "log-likelihood":
      return (activations - label)

class Activation(object):

  def __init__(self, name):
    self.name = name

  def calculate(self, z):
    if self.name == "sigmoid":
      return 1.0 / (1.0 + np.exp(-z)) #np.exp(z) returns e^z
    elif self.name == "softmax":
      return np.exp(z) / np.sum(np.exp(z))

  def derivative(self, z, j = None, i = None):
    if self.name == "sigmoid":
      return self.calculate(z) * (1.0 - self.calculate(z))
    elif self.name == "softmax":
      if j == None or i == None:
        raise TypeError("Arguments 'i' or 'j' not provided.") 
      gradient = np.array([-1.0 * self.calculate(zk)[j] * self.calculate(zk)[i]
                  if j != i else self.calculate(z)[j] * \
                  (1.0 - self.calculate(z)[j]) for zk in z])
      return gradient

class Early_Stop(object):
  """Note that all the methods in this class need to be used in conjunction
  with some kind of loop-- they need to be supplemented with some other code"""

  @staticmethod #method can be called without creating an Early_Stop object
  def GL(accuracy, stop_parameter):
    """returns "stop" if the generalization loss exceeds a parameter. If a new
    accuracy maximum has been found, this function returns "new". Otherwise,
    the function returns None"""
    local_opt = max(accuracy)
    generalization_loss = local_opt - accuracy[-1]
    to_stop = True if generalization_loss > stop_parameter else False
    if local_opt < accuracy[-1]:
      return "new"
    elif to_stop:
      return "stop"
    else:
      return None

  @staticmethod
  def average_GL(accuracy, stop_parameter, n):
    """returns "stop" if average generalization loss over the last n epochs
    exceeds a parameter. If a new accuracy maximum has been found, this function
    returns "new". Otherwise, the function returns None"""
    local_opt = max(accuracy)
    accuracy = accuracy[np.argmax(accuracy):]
    """the above line ensures that GL is only evaluated after the
    locally optimal point"""
    if len(accuracy) == 0:
      return "new"
    elif len(accuracy) >= n:
      average_gl = sum([local_opt - accuracy[-i - 1] for i in range(n)]) / \
                 len(accuracy)
      to_stop = True if average_gl > stop_parameter else False
      if to_stop:
        return "stop"
    if local_opt < accuracy[-1]:
      return "new"
    else:
      return None

  @staticmethod
  def strip_GL(accuracy, stop_parameter, k):
    """returns "stop" if generalization loss over the last k epochs
    exceeds a parameter. If a new accuracy maximum has been found, this function
    returns "new". Otherwise, the function returns None"""
    local_opt = max(accuracy)
    accuracy = accuracy[np.argmax(accuracy):]
    if len(accuracy = 0):
      return "new"
    elif len(accuracy >= k):
      strip_GL = [0 if accuracy[-i - 1] > stop_parameter else 1
                  for i in range(k)]
      to_stop = not(bool(strip_GL))
      if to_stop:
        return "stop"
    if local_opt < accuracy[-1]:
      return "new"
    else:
      return None

class Network(object): 
  
  def __init__(self, layers, cost_function = Cost("mse"),
               body_activation = Activation("sigmoid"),
               output_activation = Activation("sigmoid")):
    self.layers = layers #"layers" is a list of neurons (ex: [3, 5, 2])
    self.biases = np.array([np.random.randn(layer, 1) for layer in layers[1:]])
    """generates a random list of biases-- one array per bias
      (makes the indexing easier), and one array of arrays
      for each layer, except the first layer (which has no biases)."""
    self.weights = np.array([np.random.randn(layers[layer + 1], layers[layer])
                    for layer in range(len(layers) - 1)])
    """generates a random list of weights-- each layer l has an
    array of n arrays of length m, where m is the
    size of the layer l - 1 and n is the size of layer l."""
    self.cost = cost_function
    self.activation = body_activation
    self.output_activation = output_activation

  def feed_forward(self, a):
    #feeds an input into the network and returns its output
    for b, w in zip(self.biases, self.weights):
      a = self.activation.calculate(np.dot(w, a) + b)
    a = self.output_activation.calculate(a)
    return a
  
  def SGD(self, training_data, num_epochs, learning_rate, minibatch_size,
          validation_data = None, test_data = None, monitor = False,
          early_stopping = None, stop_parameter = None, aGL_parameter = None):
    """implements stochastic gradient descent to minimize the cost function.
    This function relies on the self.backprop function to calculate the
    gradient of the cost function, which is necessary for the updating of
    weights and biases"""
    
    if monitor or early_stopping != None:
      evaluation = {"validation accuracy": [], "validation cost": [],
                    "train accuracy": [], "train cost": []}
    if early_stopping != None:
      stored_biases = self.biases
      stored_weights = self.weights
    
    for epoch_num in range(num_epochs):
      epoch = training_data
      np.random.shuffle(epoch) #randomly shuffle epoch
      minibatches = [epoch[i:i + minibatch_size] for i in
                      range(0, len(epoch), minibatch_size)]
      """divide epoch into minibatches (the size of the minibatches is a
      hyperparameter, or a parameter not chosen by the program)"""
      
      for minibatch in minibatches:
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        """creating the (empty) arrays that will be used to store the
        gradients of the cost function for individual training examples"""
        
        for image, label in minibatch:
          delta_nabla_b, delta_nabla_w = self.backprop(image, label)
          """most of the work is done by the above line, which calculates
          the gradient of the cost function with respect to the weights
          and biases"""
          if self.cost.regularization == "L1":
            delta_nabla_w = delta_nabla_w + (self.cost.reg_parameter / len(epoch)
                                             * np.sign(self.weights))
          elif self.cost.regularization == "L2":
            delta_nabla_w = delta_nabla_w + (self.cost.reg_parameter / len(epoch) 
                                             * self.weights)       
          nabla_b = nabla_b + delta_nabla_b
          nabla_w = nabla_w + delta_nabla_w
          
        self.biases -= (learning_rate / minibatch_size) * nabla_b
        self.weights -= (learning_rate / minibatch_size) * nabla_w
      
      if test_data is None:
        print ("Epoch {0} completed".format(epoch_num + 1))
      else:
        validation_evaluate = self.evaluate_accuracy(validation_data)
        print ("Epoch {0}: {1}%".format(epoch_num + 1, validation_evaluate))
        if early_stopping or monitor:
          evaluation["validation accuracy"].append(validation_evaluate)
          evaluation["train accuracy"].append(self.evaluate_accuracy(
            training_data, is_train = True))
        if monitor:
          evaluation["validation cost"].append(self.evaluate_cost(
            validation_data))
          evaluation["train cost"].append(self.evaluate_cost(
            training_data, is_train = True))

      if early_stopping == "GL":
        to_stop = Early_Stop.GL(evaluation["validation accuracy"], stop_parameter)
        if to_stop == "stop":
          print ("End SGD: stop parameter exceeded")
          self.biases = stored_biases
          self.weights = stored_weights
          break
        elif to_stop == "new":
          stored_biases = self.biases
          stored_weights = self.weights
      elif early_stopping == "aGL":
        to_stop = Early_Stop.average_GL(evaluation["validation accuracy"],
                                        stop_parameter, aGL_parameter)
        if to_stop == "stop":
          print ("End SGD: stop parameter exceeded")
          self.biases = stored_biases
          self.weights = stored_weights
          break
        elif to_stop == "new":
          stored_biases = self.biases
          stored_weights = self.weights 

    if test_data:
      print ("Test accuracy: {0}%".format(self.evaluate_accuracy(test_data)))
      
    return evaluation

  def backprop(self, image, label):
    weighted_inputs = []
    a = image #the input (i.e., the "a"s of the first layer) is the image
    activations = [a]

    nabla_b = np.asarray([np.zeros(b.shape) for b in self.biases])
    nabla_w = np.asarray([np.zeros(w.shape) for w in self.weights])
    #creating the arrays to store the gradient of the cost function

    #Step 1: forward-propagating the data
    for b, w in zip(self.biases, self.weights):
      z = np.dot(w, a) + b
      weighted_inputs.append(z)
      a = self.activation.calculate(np.dot(w, a) + b)
      activations.append(a)

    if self.output_activation.name != self.activation.name:
      #If the output layer different from the other layers:
      activations[-1] = self.output_activation.calculate(weighted_inputs[-1])

    #print (np.array(activations).shape) --> (3,)
    #print (activations[-1].shape) --> (10, 1)
    #print (activations[-2].shape) --> (30, 1)
    
    #Step 2: computing the output error
    error = self.cost.get_error(self.output_activation, activations[-1],
                                weighted_inputs[-1], label)
    nabla_b[-1] = error
    nabla_w[-1] = np.outer(error, activations[-2])
    #print (np.array(nabla_w[-1]).shape)# --> (10, 30)
    #print (np.array(nabla_b[-1]).shape) --> (10, 1)

    #Step 3: computing the errors for the rest of the layers
    l = len(self.layers) - 2
    while l > 0:
      error = np.dot(self.weights[l].T, error) * \
             self.activation.derivative(weighted_inputs[l - 1])
      #computing the error for layer "l + 1" (index "l")
      nabla_b[l - 1] = error
      nabla_w[l - 1] = np.outer(error, activations[l - 1])
      l -= 1

    return (nabla_b, nabla_w) #returns the gradient of the cost function

  def SGD2(self, training_data, num_epochs, learning_rate, minibatch_size,
          validation_data = None, test_data = None, monitor = False,
          early_stopping = None, stop_parameter = None, aGL_parameter = None):
    """implements stochastic gradient descent to minimize the cost function.
    This function relies on the self.backprop function to calculate the
    gradient of the cost function, which is necessary for the updating of
    weights and biases"""
    
    if monitor or early_stopping != None:
      evaluation = {"validation accuracy": [], "validation cost": [],
                    "train accuracy": [], "train cost": []}
    if early_stopping != None:
      stored_biases = self.biases
      stored_weights = self.weights
    
    for epoch_num in range(num_epochs):
      epoch = training_data
      np.random.shuffle(epoch) #randomly shuffle epoch
      minibatches = [epoch[i:i + minibatch_size] for i in
                      range(0, len(epoch), minibatch_size)]
      """divide epoch into minibatches (the size of the minibatches is a
      hyperparameter, or a parameter not chosen by the program)"""
      
      for minibatch in minibatches:
        nabla_b, nabla_w = self.backprop2(minibatch)
        """most of the work is done by the above line, which calculates
        the gradient of the cost function with respect to the weights
        and biases"""
        if self.cost.regularization == "L1":
          nabla_w = nabla_w + (self.cost.reg_parameter / len(epoch)
                                           * np.sign(self.weights))
        elif self.cost.regularization == "L2":
          nabla_w = nabla_w + (self.cost.reg_parameter / len(epoch) 
                                           * self.weights)
          
        self.biases -= (learning_rate / minibatch_size) * nabla_b
        self.weights -= (learning_rate / minibatch_size) * nabla_w
      
      if test_data is None:
        print ("Epoch {0} completed".format(epoch_num + 1))
      else:
        validation_evaluate = self.evaluate_accuracy(validation_data)
        print ("Epoch {0}: {1}%".format(epoch_num + 1, validation_evaluate))
        if early_stopping or monitor:
          evaluation["validation accuracy"].append(validation_evaluate)
          evaluation["train accuracy"].append(self.evaluate_accuracy(
            training_data, is_train = True))
        if monitor:
          evaluation["validation cost"].append(self.evaluate_cost(
            validation_data))
          evaluation["train cost"].append(self.evaluate_cost(
            training_data, is_train = True))

      if early_stopping == "GL":
        to_stop = Early_Stop.GL(evaluation["validation accuracy"], stop_parameter)
        if to_stop == "stop":
          print ("End SGD: stop parameter exceeded")
          self.biases = stored_biases
          self.weights = stored_weights
          break
        elif to_stop == "new":
          stored_biases = self.biases
          stored_weights = self.weights
      elif early_stopping == "aGL":
        to_stop = Early_Stop.average_GL(evaluation["validation accuracy"],
                                        stop_parameter, aGL_parameter)
        if to_stop == "stop":
          print ("End SGD: stop parameter exceeded")
          self.biases = stored_biases
          self.weights = stored_weights
          break
        elif to_stop == "new":
          stored_biases = self.biases
          stored_weights = self.weights 

    if test_data:
      print ("Test accuracy: {0}%".format(self.evaluate_accuracy(test_data)))
      
    return evaluation
  
  def backprop2(self, minibatch):
    images = np.array([image for image, label in minibatch]).reshape(
      784, len(minibatch))
    labels = np.array([label for image, label in minibatch]).reshape(
      10, len(minibatch))
    weighted_inputs = []
    a = images #the input (i.e., the "a"s of the first layer) is the image
    activations = [a]

    nabla_b = np.array([np.zeros(b.shape) for b in self.biases])
    nabla_w = np.array([np.zeros(w.shape) for w in self.weights])
    #creating the arrays to store the gradient of the cost function

    #Step 1: forward-propagating the data
    for b, w in zip(self.biases, self.weights):
      z = np.dot(w, a) + b
      weighted_inputs.append(z)
      a = self.activation.calculate(np.dot(w, a) + b)
      activations.append(a)

    if self.output_activation.name != self.activation.name:
      #If the output layer different from the other layers:
      activations[-1] = self.output_activation.calculate(weighted_inputs[-1])

    #print (activations[-1].shape) --> (10, 10)
    #print (activations[-2].shape) --> (30, 10)
    
    #Step 2: computing the output error
    error = self.cost.get_error(self.output_activation, activations[-1],
                                weighted_inputs[-1], labels)
    nabla_b[-1] = error
    nabla_w[-1] = np.outer(error, activations[-2])
    
    #print (np.array(nabla_w[-1]).shape) --> (100, 300)
    #print (np.array(nabla_b[-1]).shape) --> (10, 10)

    #Step 3: computing the errors for the rest of the layers
    l = len(self.layers) - 2
    while l > 0:
      error = np.dot(self.weights[l].T, error) * \
             self.activation.derivative(weighted_inputs[l - 1])
      #computing the error for layer "l + 1" (index "l")
      nabla_b[l - 1] = error
      nabla_w[l - 1] = np.outer(error, activations[l - 1])
      l -= 1

    return (nabla_b, nabla_w) #returns the gradient of the cost function

  def evaluate_accuracy(self, test_data, is_train = False):
    #returns number correct when the network is evaluated using test data
    test_results = [(np.argmax(self.feed_forward(image)), label)
                        for (image, label) in test_data]
    if is_train:
      return round((sum(int(image == np.argmax(label)) for (image, label) in
                 test_results) / len(test_data) * 100.0), 2)
    else:
      return round((sum(int(image == label) for (image, label) in test_results) \
             / len(test_data) * 100.0), 2)

  def evaluate_cost(self, test_data, is_train = False):
    #returns cost when the network is evaluated using test data
    if is_train:
      return self.cost.calculate([(self.feed_forward(image), label)
                                  for (image, label) in test_data])
    else:
      return self.cost.calculate([(self.feed_forward(
        image), mnist.vectorized_result(label)) for (image, label) in test_data])


#Main
def main(structure, learning_rate, minibatch_size, num_epochs,
         cost_function = Cost("mse"),
         body_activation = Activation("sigmoid"),
         output_activation = Activation("sigmoid"), monitor = False):
  data = mnist.load_data()

  digit_classifier = Network(structure, cost_function = cost_function,
                             body_activation = body_activation,
                             output_activation = output_activation)
    
  print ("Evaluation without training: {0}%".format(
    digit_classifier.evaluate_accuracy(data["test"])))
  
  print ("""Learning rate: {0}\nMinibatch size: {1}\
        \nNumber of epochs: {2}\nStructure: {3}\nCost function: {4}\nBody activation function: {5}\nOutput activation function: {6}"""
         .format(learning_rate, minibatch_size, num_epochs,
          digit_classifier.layers, digit_classifier.cost.name,
          digit_classifier.activation.name,
          digit_classifier.output_activation.name))
  print ("Training in process...")
  
  accuracy = digit_classifier.SGD(data["train"], num_epochs, learning_rate,
                                  minibatch_size, validation_data =
                                  data["validation"], test_data = data["test"],
                                  monitor = monitor)

  return digit_classifier

#Testing area
if __name__ == "__main__":
  main([784, 30, 10], 0.5, 10, 30, cost_function = Cost("cross-entropy"),
       monitor = False)
