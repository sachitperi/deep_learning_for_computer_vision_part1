# import the necessary packages
import numpy as np

class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
        # initialize the list of weights matrices, then store thee network architecture and learning
        self.W = []
        self.layers = layers
        self.alpha = alpha

        # start looping from the index of the first layer but stop before we reach the last two layers
        for i in np.arange(0, len(layers)-2):
            # randomly initialize a weight matrix connecting the number of nodes in each respective layer together, adding an extra node for the bias
            w = np.random.randn(layers[i] + 1, layers[i+1] + 1)
            self.W.append(w/np.sqrt(layers[i]))

        # the last  two layer are a special case where the input connections need a bias term but the output does not
        w = np.random.randn(layers[-2]+1, layers[-1])
        self.W.append(w/np.sqrt(layers[-2]))

    def __repr__(self):
        # construct and return a string that represents the network architecture
        return "NeuralNetwork: {}".format("-".join(str(l) for l in self.layers))

    def sigmoid(self, x):
        # compute and return the sigmoid activation value for a given input value
        return 1.0/(1+np.exp(-x))

    def sigmoid_deriv(self, x):
        # compute the derivative of sigmoid function ASSUMING that 'x' has already been passed through the 'sigmoid' function
        return x * (1-x)

    def fit(self, X, y, epochs=1000, displayUpdate=100):
        """
        The method requires two parameters followed by two opttional one
        :param X: Training data
        :param y: Class Labels
        :param epochs: number of iterations for which we train the ntework for
        :param displayUpdate: controls how many N epochs we'll train our network for
        """
        # insert a column of 1's as the last entry in the feature matrix -- this little trick allows us to treat the bias as a trianable parameter within the weight matrix
        X = np.c_[X, np.ones((X.shape[0]))]

        # loop over the desired number of epochs
        for epoch in np.arange(0, epochs):
            # loop overr each individual data point and train the network
            for (x, target) in zip(X, y):
                self.fit_partial(x, target)

            # check to see if we should display a training update
            if epoch == 0 or (epoch+1)% displayUpdate == 0:
                loss = self.calculate_loss(X, y)
                print("[INFO] epoch={}, loss={:.7f}".format(epoch+1, loss))

    # actual heart of the backpropagation algorithm is found under the fit_partial method
    def fit_partial(self, x, y):
        """
        The function requires two parameters
        :param x: An individual datapoint from our design matrix
        :param y: The corresponding class label
        :return: A list responsible for storing output activations as our datapoint x forward propagates through the network.
        we initialize the list with x which is simply the input datapoint.
        """
        # construct our list of activations for each layer as our datapoint flows through the network;
        # the first activation is a special case -- its just the input feature vector itself
        A = [np.atleast_2d(x)]

        # FEEDFORWARD
        # loop over the layers in the network
        for layer in np.arange(0, len(self.W)):
            # feed forward the activation in the current layer by taking a dot product between the activation and the weight matrix
            # -- this is called the "net input" to the current layer
            net = A[layer].dot(self.W[layer])

            # computing the "net output" is simply applying our non linear activation function to the net input
            out = self.sigmoid(net)

            # once we have the net output, add it to our list of activations
            A.append(out)

        # BACKPROPAGATION
        # the first phase of backpropagation is to compute the differencee between our prediction (the final output in the activation list) and true target value
        error = A[-1] - y

        # from here we need to apply chain rule and apply our list of deltas 'D';
        # the first entry in the deltas is simply the error of the output layer times the derivative of our activation function
        D = [error*self.sigmoid_deriv(A[-1])]

        # once you apply chain rule it becomes super easy to implement with a for loop -- simply loop over the layers in reverse order
        # (ignoring the last two  since we already have taken them into account)
        for layer in np.arange(len(A) - 2, 0, -1):
            # the delta for the current layer, followed by multiplying the delta for the activations of the current layer
            delta = D[-1].dot(self.W[layer].T)
            delta = delta*self.sigmoid_deriv(A[layer])
            D.append(delta)

        # since we looped over the layers in reverse order we need to reverse the deltas
        D = D[::-1]

        # WEIGHT UPDATE PHASE
        # loop over the layers
        for layer in np.arange(0, len(self.W)):
            # update our weights by taking the dot product of the layer activations with their respective deltas,
            # then multiplying this value by some learning rate and adding to our weight matrix
            # -- this is where the actual learning takes place
            self.W[layer] += -self.alpha*A[layer].T.dot(D[layer])

    def predict(self, X, addBias=True):
        # initialize the output prediction as the input feature
        # -- this value will be (forward) propagated through the network to obtain the final predictions
        p = np.atleast_2d(X)

        # check to see if bias colum needs to be added
        if addBias:
            # add a column of 1's as the last entry of the feature matrix (bias)
            p = np.c_[p, np.ones((p.shape[0]))]

        # loop over our layers in the network
        for layer in  np.arange(0, len(self.W)):
            # computing the output prediction is as simple as taking the dot product between the current activation value 'p'
            # and the weight matrix associated with the current layer, then passing this value through a non linear activation function
            p = self.sigmoid(np.dot(p, self.W[layer]))

        # return the predicted value
        return p

    def calculate_loss(self, X, targets):
        # make predictions for the input datapoints and then compute the loss
        targets = np.atleast_2d(targets)
        predictions = self.predict(X, addBias=False)
        loss = 0.5*np.sum((predictions - targets)**2)

        # return the loss
        return loss