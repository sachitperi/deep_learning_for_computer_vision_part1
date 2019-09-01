# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from utilities.nn import ShallowNet
from keras.optimizers import SGD
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np

# loading the training and testing data, then scale it into range [0, 1]
print("[INFO] Loading the data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float")/255.0
testX = testX.astype("float")/255.0

# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

# initialize the label names for CIFAR10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer",
              "dog", "frog", "horse", "ship", "truck"]

# initialize the optimizer model
print("[INFO] compiling modeel...")
opt = SGD(lr=0.003)
model = ShallowNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# training the network
print("[INFO] training the network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
              batch_size=32, epochs=40, verbose=1)

# evaluate the network
print("[INFO] evaluating the network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))

# plot the loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 40), H.history["loss"], label="training loss")
plt.plot(np.arange(0, 40), H.history["val_loss"], label="validation loss")
plt.plot(np.arange(0, 40), H.history["acc"], label="training accuracy")
plt.plot(np.arange(0, 40), H.history["val_acc"], label="validation accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig("shallownet_cifar10.png")
plt.show()