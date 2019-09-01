# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from utilities.preprocessing import ImageToArrayPreprocessor
from utilities.preprocessing import SimplePreprocessor
from utilities.datasets import SimpleDatasetLoader
from utilities.nn import ShallowNet
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse

# construct an argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to the input dataset")
# path where we would like to save the network after training is complete
ap.add_argument("-m", "--model", required=True,
                help="path to the output model")
args = vars(ap.parse_args())

# grab the list of images we'll be describing
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

# initialize the image preprocessors
sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()

# load the dataset from disk then scale the raw pixel intensities to the range [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, label) = sdl.load(imagePaths, verbose=500)
data = data.astype("float")/255.0

# partition the data into training and testing splits using 75% for the training and remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, label,
                                                  test_size=0.25, random_state=42)

# convert the labels from integers to vectors
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

# initialize the optimizer module
print("[INFO] compiling the model...")
opt = SGD(lr=0.0009)
model = ShallowNet.build(width=32, height=32, depth=3, classes=3)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])
print(model.summary)

# train the network
print("[INFO] training the network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
              batch_size=32, epochs=100, verbose=1)

# save the network to disk
print("[INFO] serializing the model...")
model.save(args["model"])

# evaluate the network
print("[INFO] Evaluating the network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1),
                            target_names=["cat", "dog", "panda"]))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="training loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="validation loss")
plt.plot(np.arange(0, 100), H.history["acc"], label="training accuracy")
plt.plot(np.arange(0, 100), H.history["val_acc"], label="validation accuracy")
plt.title("Training loss and accuracy")
plt.xlabel("Epoch #")
plt.ylabel("loss/acc")
plt.legend()
plt.savefig("shallownet_animals.png")
plt.show()