import keras

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ['KMP_DUPLICATE_LIB_OK'] = "True"
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

(x, y), (te_x, te_y) = keras.datasets.mnist.load_data()

x = (x / 255).astype("float32").reshape(x.shape[0], -1)
te_x = (te_x / 255).astype("float32").reshape(te_x.shape[0], -1)

y = keras.utils.to_categorical(y, 10).astype("float32")
te_y = keras.utils.to_categorical(te_y, 10).astype("float32")

model = keras.Sequential()
model.add(keras.layers.Dense(100, input_shape=(784,), activation="sigmoid"))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(10, input_shape=(100,), activation="softmax"))

model.compile(optimizer = keras.optimizers.SGD(lr=0.1), loss = "categorical_crossentropy",
              metrics = ["categorical_accuracy"])
model.fit(x, y, batch_size=10, epochs=30, verbose=2, validation_split=0.2)

model.evaluate(te_x, te_y, verbose = 2)