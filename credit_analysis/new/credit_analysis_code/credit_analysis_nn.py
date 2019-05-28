"""

"credit_analysis_nn.py"

A program that uses keras to build and train a neural network to grade
the quality of a loan.

"""

#Libraries
import pandas as pd
import xlrd as xl
import numpy as np
import keras

#Data processing
def sigmoid_normalize(raw_array, range_ = None):
  #function that converts a list of values between any range to [0, 1]
  array = np.copy(raw_array).astype(np.float32)
  if range_ == None:
    range_ = (min(array), max(array))
  if range_ == (0, 1):
    return array
  #Step 1: subtract minimum from everything
  array -= range_[0]
  #Step 2: divide by range
  dist = abs(range_[0]) + abs(range_[1])
  array /= dist
  return np.nan_to_num(array)

def convert_categorical(categoricals, range_):
  #converts a list of categorical variables to an integer list, range = [0, 1]
  to_int = len(range_)
  fractions = np.array([i / (to_int - 1)  for i in range(to_int)],
                       dtype = np.float32)
  return np.nan_to_num(np.array([fractions[range_.index(categorical)]
          for categorical in categoricals], dtype = np.float32))

def to_int(n):
  #turns every element in a list into an int
  fin = []
  for element in n:
    try:
      fin.append(int(element))
    except ValueError:
      fin.append(0)
  return np.nan_to_num(np.array(fin, dtype = np.float32))

def vectorize(value, range_):
  #takes a value and vectorizes it (one-hot encoder)
  result = np.zeros((len(range_), ), dtype = np.float32)
  result[range_.index(value)] = 1.0
  return result

def strip(n):
  #strips a list of strings of everything but digits and decimals
  if isinstance(n, str):
    return "".join(ch for ch in str(n) if str(ch).isdigit() or str(ch) == ".")
  else:
    return ["".join(ch for ch in str(s) if str(ch).isdigit() or str(ch) == ".")
            for s in n]

def get_range(data):
  #gets the ranges for a list
  ranges = []
  for element in data:
    if element in ranges:
      continue
    else:
      ranges.append(element)
  return ranges

def unison_shuffle(a, b):
  #returns unison shuffled copies of two np.arrays (not in-place)
  p = np.random.permutation(len(a))
  return a[p], b[p]

def load_file(filestream):
  #reads a specific excel file and prepares it for data processing
  data = pd.read_excel(filestream)
  del data["sub_grade"]
  del data["loan_status"]

  labels = []
  range_ = get_range(data["grade"])
  for label in np.asarray(data["grade"]):
    labels.append(vectorize(label, range_))
  
  del data["grade"]

  for feature in data.columns:
    if feature == "term" or feature == "emp_length":
      data[feature] = to_int(strip(data[feature]))
    try:
      data[feature] = sigmoid_normalize(data[feature])
    except ValueError:
      data[feature] = convert_categorical(data[feature],
                                          get_range(data[feature]))

  return (data.values.reshape(len(data.index), len(data.columns), 1),
          np.array(labels), data.columns)
 
def load_data(ratio, keras_ = True):
  #data processer (essentially a wrapper for "load_file()")
  inputs, labels, cols = load_file("/Users/ryan/Documents/Coding/neural_networks/credit_analysis/credit_analysis_dataset/credit_analysis_dataset.xlsx")
  
  if not keras_:
    big_data = np.array(list(zip(inputs, labels)))
    num_train = int(ratio * len(big_data))
    num_validation = int((len(big_data) - num_train) / 2)

    data = {"train": np.array(big_data[0:num_train]),
            "validation": np.array(big_data[num_train:num_validation + num_train]),
            "test": np.array(big_data[num_validation + num_train:])}
  else:
    big_data = unison_shuffle(inputs.reshape(len(inputs), len(inputs[0])),
                              labels.reshape(len(labels), len(labels[0])))
    num_train = int(ratio * len(big_data[0]))
    num_validation = int((len(big_data[0]) - num_train) / 2)
    
    data = {"train": (big_data[0][:num_train], big_data[1][:num_train]),
            "validation": (big_data[0][num_train:num_validation + num_train],
                           big_data[1][num_train:num_validation + num_train]),
            "test": (big_data[0][num_validation + num_train:],
                     big_data[1][num_validation + num_train:])}  

  return (data, cols, big_data)

def load_old(ratio, keras_ = True):
  #loads the processed data from the previous dataset (hopefully not used!)
  inputs = []
  labels = []
  with open("/Users/ryan/Documents/Coding/neural_networks/credit_analysis/credit_analysis_results/credit_analysis_processed_data.txt",
            "r") as filestream:
    counter = 1
    temp = []
    for line in filestream:
      if counter % 15 == 0:
        temp.append(float(strip(line)))
        inputs.append(np.array(temp, dtype = np.float32).reshape(len(temp), 1))
        temp = []
      elif counter % 16 == 0:
        labels.append(np.array(float(strip(line)), dtype = np.float32))
        counter = 0
      else:
        temp.append(float(strip(line)))
      counter += 1
      
  inputs = np.array(inputs, dtype = np.float_)
  labels = np.array(labels, dtype = np.float_).reshape(len(labels), 1)
      
  if not keras_:
    big_data = np.array(list(zip(inputs, labels)))
    num_train = int(ratio * len(big_data))
    num_validation = int((len(big_data) - num_train) / 2)
    
    data = {"train": np.array(big_data[0:num_train]),
            "validation": np.array(big_data[num_train:num_validation + num_train]),
            "test": np.array(big_data[num_validation + num_train:])}
    
  else:
    big_data = unison_shuffle(np.array(inputs, dtype = np.float32).
                              reshape(len(inputs), len(inputs[0])),
                              np.array(labels, dtype = np.float32).
                              reshape(len(labels), len(labels[0])))
    num_train = int(ratio * len(big_data[0]))
    num_validation = int((len(big_data[0]) - num_train) / 2)
    
    data = {"train": (big_data[0][:num_train], big_data[1][:num_train]),
            "validation": (big_data[0][num_train:num_validation + num_train],
                           big_data[1][num_train:num_validation + num_train]),
            "test": (big_data[0][num_validation:], big_data[1][num_validation:])}

  return (data, 0, big_data)

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
            epochs = 100, batch_size = 20, verbose = 2,
            validation_data = data["validation"])
  
  return model

def evaluate(data, model):
  te_X, te_Y = data["test"]

  evaluation = model.evaluate(te_X, te_Y, verbose = 0)
  print ("Accuracy: {0}%".format(round(evaluation[1] * 100, 2)))

  return evaluation

#Testing area
if __name__ == "__main__":
  data, cols, big_data = load_data(0.8, keras_ = True)

  print ("Starting training")
  model = train(data, build())
  evaluation = evaluate(data, model)
