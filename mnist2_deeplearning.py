from tensorflow import keras

from tensorflow.keras.datasets import fashion_mnist
(train_images,train_lables),(test_images,test_lables)=fashion_mnist.load_data()

from tensorflow.keras import Sequential , layers

# model = Sequential()
# model.add(layers.Dense(32,activation="relu"),input_shape=(784,))
# model.add(layers.Dense(64,activation="relu"))
# model.add()
# model.add()

model = Sequential([
    layers.Dense(32,activation="relu"),
    layers.Dense(64,activation="relu"),
    layers.Dense(128,activation="relu"),
    layers.Dense(10,activation="softmax")
])

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255

train_lables

model.compile(optimizer="rmsprop",loss="sparse_categorical_crossentropy",metrics=["accuracy"])

model.fit(train_images, train_lables, epochs=10, batch_size=128)

from PIL import Image
import numpy as np

img = Image.open("/content/4.png")
img = img.convert("L")
img = np.array(img)
img = img.reshape(1,784)

pred = model.predict(img)

np.argmax(pred)

pred.sum()

def ifi(output):
  if output == 0:
    return "ya apka result hai "
  elif output == 1:
    return "ya 1 hai"
  elif output == 2:
    return "ya two hai"
  elif output == 3:
    return "ya 3 hai"
  elif output == 4:
    return "this ia four"
  elif output == 5:
    return "this is five"
  elif output == 6:
    return "this is 6"
  elif output == 7:
    return "ya 7 hai"
  elif output == 8:
    return "8"
  elif output == 9:
    return "nine"

def image_predictor(img):
  img = Image.open(img)
  img = img.convert("L")
  img = np.array(img)
  img = img.reshape(1,784)
  output = np.argmax(model.predict(img))
  return ifi(output)

image_predictor("0.png")

model.save("mnist.keras")

