"""

"data_loader.py"

A program to load the MNIST dataset as numpy arrays. You can download
the datasets at "http://yann.lecun.com/exdb/mnist/".

The format for the MNIST datasets are as follows:

TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
[offset] [type]          [value]          [description] 
0000     32 bit integer  0x00000801(2049) magic number (MSB first) 
0004     32 bit integer  60000            number of items 
0008     unsigned byte   ??               label 
0009     unsigned byte   ??               label
........ 
xxxx     unsigned byte   ??               label

The labels values are 0 to 9.

TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
[offset] [type]          [value]          [description] 
0000     32 bit integer  0x00000803(2051) magic number 
0004     32 bit integer  60000            number of images 
0008     32 bit integer  28               number of rows 
0012     32 bit integer  28               number of columns 
0016     unsigned byte   ??               pixel 
0017     unsigned byte   ??               pixel 
........ 
xxxx     unsigned byte   ??               pixel

Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

TEST SET LABEL FILE (t10k-labels-idx1-ubyte):
[offset] [type]          [value]          [description] 
0000     32 bit integer  0x00000801(2049) magic number (MSB first) 
0004     32 bit integer  10000            number of items 
0008     unsigned byte   ??               label 
0009     unsigned byte   ??               label 
........ 
xxxx     unsigned byte   ??               label

The labels values are 0 to 9.

TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
[offset] [type]          [value]          [description] 
0000     32 bit integer  0x00000803(2051) magic number 
0004     32 bit integer  10000            number of images 
0008     32 bit integer  28               number of rows 
0012     32 bit integer  28               number of columns 
0016     unsigned byte   ??               pixel 
0017     unsigned byte   ??               pixel 
........ 
xxxx     unsigned byte   ??               pixel

The images are 28 x 28 pixels, with each pixel being represented by a number between 0 and 255. There are 70000 images
in total, with 60000 being the training/validation data and 10000 being the testing data.

"""

# Libraries
import matplotlib.pyplot as plt # for displaying images
import codecs # for converting from binary to integers
import numpy as np # parsing the files
import gzip # for unzipping files

def to_int(b):
  # function that takes in a byte array and returns an integer
  return int(codecs.encode(b, "hex"), 16)

def normalize(raw_array, range_):
  # function that converts a list of values between any range to [0, 1]
  array = np.copy(raw_array).astype(np.float32) # raw_array is not writeable
  if range_ == (0, 1): return array
  # Step 1: subtract minimum from everything
  array -= range_[0]
  # Step 2: divide by range
  dist = abs(range_[0]) + abs(range_[1])
  array /= dist
  return array

def vectorize(num):
  """function that takes in a number and vectorizes it.
  For example, if the input is a 7, this function will return [[0.]
                                                               [0.]
                                                               [0.]
                                                               [0.]
                                                               [0.]
                                                               [0.]
                                                               [0.]
                                                               [1.]
                                                               [0.]
                                                               [0.]]"""
  result = np.zeros((10, 1))
  result[num] = 1.0
  return result

def load_file(file, mode):
  # function that loads a specific zipped file
  
  with gzip.open(file, "rb") as raw:
    data = raw.read()
    magic_number = to_int(data[:4])
    # the first four items in the file make up the magic number, which identifies the file as images or labels"""
    length = to_int(data[4:8])
    """the next four items indicate the length of the file, so the training files have a length of 60000, and the
     testing files will have a length of 10000"""
    if magic_number == 2049: # 2049 is the magic number for labels
      parsed = np.frombuffer(data, dtype = np.uint8, offset = 8)
      """almost all of the work is done by the line above. In essence, the line above is converting the file from byte 
      array to a re-shaped numpy array with dimensions (60000,). (Note the difference between (60000,) and (60000, 1): 
      an array of shape (60000,) is a 1-D array of length 60000, while an array of shape (60000, 1) is a 60000-D array 
      with each dimension of length 1"""
      
    elif magic_number == 2051: # 2051 is the magic number for images
      num_rows = to_int(data[8:12])
      # the 8th through 12th items in the file give the number of rows in one image
      num_columns = to_int(data[12:16])
      # the next four items give the number of columns in one image
      if mode == "mlp":
        parsed = normalize(np.frombuffer(data, dtype = np.uint8, offset = 16).
                           reshape(length, num_rows * num_columns, 1), (0, 255))
      else:
        parsed = normalize(np.frombuffer(data, dtype = np.uint8, offset = 16).
                         reshape(length, num_rows, num_columns), (0, 255))
      # converting the file from byte array to reshaped numpy array in order to prepare it for usage
    else:
      parsed = -1 # something went wrong

    return parsed

def load_data(mode):
  # wrapper function that implements load_file() to parse all of the MNIST files
  data = {"train": [], "validation": [], "test": []}
  
  train_images = load_file("/Users/Ryan/PycharmProjects/neural-networks/mnist/dataset/train-images-idx3-ubyte.gz",
                           mode = mode)
  train_labels = load_file("/Users/Ryan/PycharmProjects/neural-networks/mnist/dataset/train-labels-idx1-ubyte.gz",
                           mode = mode)
  data["validation"] = np.asarray(list(zip(train_images[:10000],
                                       np.asarray(train_labels))))
  """data["validation"] is a set of 10,000 tuples (x, y) containing the
  28 x 28 image "x" and the corresponding non-vectorized label "y" """
  data["train"] = np.asarray(list(zip(train_images[10000:],
                                  np.asarray([vectorize(i) for
                                              i in train_labels[10000:]]))))
  """data["train"] is a set of 50,000 tuples (x, y) containing the
  28 x 28 image "x" and the corresponding 10-D vectorized label "y" """

  test_images = load_file("/Users/Ryan/PycharmProjects/neural-networks/mnist/dataset/t10k-images-idx3-ubyte.gz",
                          mode = mode)
  test_labels = load_file("/Users/Ryan/PycharmProjects/neural-networks/mnist/dataset/t10k-labels-idx1-ubyte.gz",
                          mode = mode)
  data["test"] = np.asarray(list(zip(test_images,
                                 np.asarray(test_labels))))

  return data

def display_image(pixels, label = None):
  # function that displays an image using matplotlib-- not really necessary for the digit classifier
  figure = plt.gcf()
  figure.canvas.set_window_title("Number display")
  
  if label: plt.title("Label: \"{label}\"".format(label = label))
  else: plt.title("No label")
    
  plt.imshow(pixels, cmap = "gray")
  plt.show()

# Testing area
if __name__ == "__main__":
  data = load_data("conv")
  epoch = data["train"]
  # np.random.shuffle(epoch) # randomly shuffle epoch
  minibatches = [epoch[i:i + 16] for i in
                        range(0, len(epoch), 16)]
  for minibatch in minibatches:
    for image, label in minibatch:
      display_image(image, label = np.argmax(label))
      break
    break
