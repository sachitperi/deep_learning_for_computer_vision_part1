# import the necessary packages
import numpy as np

class Perceptron:
    """
    Define the constructor of the class which accepts single required parameter followed by an optional one
    1) N: The no of column in our input feature vectors.
    2) alpha: Learning rate for the perceptron algorithm. We will set this value to 0.1 by default. Common choices of learning rates are alpha=0.1, 0.01, 0.001
    """
    def __init__(self, N, alpha=0.1):
        # initialize the weight matrix and store the learning rate
        # The weight matrix will have N=1 entries one for each of the N imputs in the feature vector, plus one for the bias vector
        self.W = np.random.randn(N+1)/np.sqrt(N)
        self.alpha = alpha

    def step(self, x):
        # apply thr steep function
        return 1 if x > 0  else 0
    """
    The fit method requires two parameters followed by one single optional parameter 
    1) X: Actual training data
    2) y: target output class
    3) epochs: the number of epochs the perceptron will train for 
    """
    def fit(self, X, y, epochs=10):
        # insert a column of 1s as the last entrry of the feature matrix -- this little trick allows us to treat thee bias as a trainablee parameter within the weight box
        X = np.c_[X, np.ones((X.shape[0]))]

        # loop over the deesired number of epochs
        for epoch in np.arange(0, epochs):
            # loop over each individual data point
            for (x, target) in zip(X, y):
                # take the dot product between the input features and the weight matrix, then pass this value through the step function to obtain the prediction
                p = self.step(np.dot(x, self.W))

                # only perfor an weight update if our prediction doesnot match the target
                if p != target:
                    # determine the error
                    error = p - target

                    # update the weight matrix
                    self.W += -self.alpha * error * x

    def predict(self, X, addBias=True):
        # ensure our input matrix
        X = np.atleast_2d(X)

        # check to see the bias column should be added
        if addBias:
            # insert a column of 1's as the last entry in the feature matrix (bias)
            X = np.c_[X, np.ones((X.shape[0]))]

        # take the dot product between the input features and the weight matrix, then pass the value through the step function
        return self.step(np.dot(X, self.W))