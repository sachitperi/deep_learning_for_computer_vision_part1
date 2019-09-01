### Objective
* Fundamentals of neural network in depth 
    * Discussion of artificial neural networks
    * Classic perceptron algo
    * backpropagation and its implimentation in python from scratch 
    * Standard feedforward nn using keras 
    * Four ingredients needed when building any neural network 
    
#### Neural Network Basics 
* Artificial Neural Network and their relation to biology 
* The seminal Perceptron algorithm 
* The backpropagation algorithm and how it can be used to train multi-layer neural network 
* How to train nn using Keras 

Basics include 
* architecture
* node, types 
* aglorithms for teaching the nn  

###### Implimenting the perceptron
```
--- utilities
|   |--- __init__.py
|   |--- nn
|   |   |--- __init__.py
|   |   |--- perceptron.py
``` 

####### output of perceptron on peerceptron_or
```
python3 perceptron_or.py 
[INFO] training perceptron...
[INFO] testing perceptron...
[INFO] data=[0 0], ground-truth=0, pred=0
[INFO] data=[0 1], ground-truth=1, pred=1
[INFO] data=[1 0], ground-truth=1, pred=1
[INFO] data=[1 1], ground-truth=1, pred=1
```

####### output of perceptron on peerceptron_and
```
python3 perceptron_and.py 
[INFO] training perceptron...
[INFO] testing perceptron...
[INFO] data=[0 0], ground-truth=1, pred=0
[INFO] data=[0 1], ground-truth=0, pred=0
[INFO] data=[1 0], ground-truth=0, pred=1
[INFO] data=[1 1], ground-truth=1, pred=1
```

####### output of perceptron on peerceptron_xor
```
python3 perceptron_xor.py 
[INFO] training perceptron...
[INFO] testing perceptron...
[INFO] data=[0 0], ground-truth=0, prred=1
[INFO] data=[0 1], ground-truth=1, prred=0
[INFO] data=[1 0], ground-truth=1, prred=0
[INFO] data=[1 1], ground-truth=0, prred=0
```
* No mater how may times you run this experimeent with varying learning rates 
or different weight initialization schemes, you will never be able to correctly 
model the XOR function with a single layer perceptron. Instead we need more 
layers and this is the starting point for deep learning.

#### Backpropagation and Multi-layer Networks 

##### Backpropagation Summary 

#### Multilayerd Networks with Keras 
Fow to impliment feed forward neural networks using keras and apply them to 
MNIST and CFAIR-10 datasets. The results are not state of the art but will 
serve the following two purposes
* to demonstrate how to impliment standard neural networks using keras
* to obtain a baseline result using standard neuralnetwork and later 
compare it with that of convolutional neuralnetwork

#####MNIST 

##### CFAIR-10
A collection of 60000, 32 X 32 RGB images. 
Each image in the dataset is represented by 32 x 32 x 3 = 3072 integers. 
CFAIR-10 consists of 10 classes 
`airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck`.  
Each class is evenly represented with 6000 images per class. 
CFAIR-10 is substantially harder than MNIST dataset. The challenge comes 
from dramatic variance in how objects appear.

#### The four ingredients in a Neural Network Recipe  
There are four main ingredients you need to put together your 
own neural network and deep learning algorithm 
* dataset 
* model/architecture 
* loss function 
* optimization method 

### Summary
* Reviewed the fundamentals of neural network 
* Studied the Perceptron Algorithm. Perceptron ha one major flaw, 
it accurately classifies nonlinear separable points. In order to work with 
more challenging datasets need both 
`a) non linear activation function and b) multi-layer networks` 
* To train the multilayer networks we must use backprop algorithm
* Implementation of backprop 
and its demonstration when training the multi-layered network 
with non-linear activations, can model non-linearly separable datasets such as XOR 
* Keras. Using existing implementation of backprop in keras
* four key ingredients to neeural network and deep learning 