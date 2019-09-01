# importing the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import argparse

# construct an argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
                help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# grab the MNIST dataset (if it is being downloaded for the first time, the download may take a minute -- the 55MB MNIST dataset will be downloaded)
dataset = datasets.fetch_mldata("MNIST Original")

# scale the raw pixel intensities to the range [0, 1.0] then construct the training and testing splits
data = dataset.data.astype("float")/255.0
(trainX, testX, trainY, testY) = train_test_split(data,
                                                  dataset.target, test_size=0.25 )
# construct a label binarizer
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

# define the 784-256-128-10 architecture
model = Sequential()
model.add(Dense(256, input_shape=(784, ), activation="sigmoid"))
model.add(Dense(128, activation="sigmoid"))
model.add(Dense(10, activation="softmax"))

# train the model using sgd
print(("[INFO] Training network..."))
sgd = SGD(0.01)
model.compile(loss="categorical_crossentropy", optimizer=sgd,
              metrics=["accuracy"])
H = model.fit(trainX, trainY, validation_data=(testX, testY),
              epochs=200, batch_size=128)

# evaluate the network
print("[INFO] evaluating the network")
predictions = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1),
                            target_names=[str(x) for x in lb.classes_]))
# plot the loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 200), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 200), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 200), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 200), H.history["val_acc"], label="val_acc")
plt.title("Training loss and accuracy")
plt.xlabel("EPOCH #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])