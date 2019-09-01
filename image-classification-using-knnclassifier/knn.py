# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from utilities.preprocessing import SimplePreprocessor
from utilities.datasets import SimpleDatasetLoader
from imutils import paths
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to the input dataset")
ap.add_argument("-k", "--neighbors", type=int, default=1,
                help="# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1,
                help="# of jobs for k-NN distance (-1 uses all available cores)")
args = vars(ap.parse_args())

"""
To run the script we require one command line followed by two optional ones
* --dataset : The path to where our input image dataset resides on disk.
* --neighbors : Optional, the number of neighbors k to apply when using k-NN algorithm.
* --jobs : Optional, the number of concurrent jobs to run when computing the distance between an input data point and the training set. 
A value of -1 will use all available cores on the processor
"""

# After receiving the command line args next steps is to grab the file paths of the images in our dataset followed by loading and preprocessing them
# grab the list of that we will be describing
print("[INFO] loading images ...")
imagePaths = list(paths.list_images(args["dataset"]))

# initialize the image preprocessor, load the dataset from disk, and reshape the data matrix
sp = SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.reshape((data.shape[0], 3072))

# show some information on memory consumption of the images
print("[INFO] feature matrix: {:.1f}MB".format(data.nbytes/(1024*1000.0)))

# encode the labels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)

# Partition the data into training and testing splits using 75% for training and remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                  test_size=0.25, random_state=42)
# train and evaluate a knn classifier on raw pixel intensities
print("[INFO] evaluating a k-NN classifier")
model = KNeighborsClassifier(n_neighbors=args["neighbors"],
                             n_jobs=args["jobs"])
model.fit(trainX, trainY)
print(classification_report(testY, model.predict(testX),
                            target_names=le.classes_))
