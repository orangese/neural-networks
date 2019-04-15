
"""

"mnist_test.py"

Program to test mnist_loader2.py and mnist_loader.py

"""

import mnist_loader2
import mnist_loader
import numpy as np

mnist_loader_data = mnist_loader.load_data()
mnist_loader2_data = mnist_loader2.load_data()

##for image, label in mnist_loader_data["train"][:100]:
##  for image2, label2 in mnist_loader2_data["train"][:100]:
##    if np.array_equal(image2, image):
##      print ("one")
##print ('done')

for pixel in mnist_loader_data["train"][0][0]:
  if pixel != 0:
    print (pixel)
    break

for pixel in mnist_loader2_data["train"][0][0]:
  if pixel != 0:
    print (pixel)
    break
