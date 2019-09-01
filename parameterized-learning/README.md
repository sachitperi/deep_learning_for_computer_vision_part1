### Parameterized Learning

A machine learninf model that can learn patterns from our input data during training time (requiring us to spend more time on training process), but have the benefit of being defined by a small number of parameters that can easily be used to represent the model, regardless of training size.  
`"A Learning modeel that sumarizes data with a set of parameters of fixed size (independent of number of training examples) is called a parametric model. No mater how much data you throw at a parametric model, it wont change it's mind about how many parameters it needs."` - `Russel and Norvig (2009)`  
Also please refer to the notes of Andrej Karpathy's excellent Linear Classification notes inside Standford's cs231 in class. A big thank you to Karpathy and the rest of the cs231's teaching assistants for putting together such accessible notes.  

#### Four Components of Parameterized Learning
Paremeterization is the process of defining the necessary parameters of a given model. 
In the task of machine learning parameterization involves probelem in terms of four key components. 
1) data
2) a scoring function
3) a loss function
4) weights and biases

##### Data
The input data that we are going to learn from. 
Includes both the data points (raw pixel intensities from images, 
extracted features, etc) and their associated class labels.

##### Scoring Function
The scoring function accepts our data as input and maps the data to class labels.  
`INPUT_IMAGES => F(INPUT_IMAGES) => OUTPUT_CLASS_LABELS`  

##### Loss Function
A loss function quantifies how well our predicted class labels agree with our ground-truth labels. 
The higher the level of agreement between these two sets of labels, the lower our loss (and higher our classification accuracy atleast on our training set).
Our goal while training machine learning model is to minimize the loss function.

##### Weights and Biases
The weight matrix typically denoted as `W` and the bias vector `b` are 
called `weights` or `parameters` of our classifier that we will be 
optimizing. Based on the output of the scoring function and loss function 
we will be tweaking and fiddling with the values of the weights and biases 
to increase classification accuracy.

### Linear classification: From images tto labels
Dataset:- Animals dataset (dogs cats and pandas)
```
Total no of images N: 3000
Each individual image D: 32 x 32 x 3 = 3072 (pixels)
Class Labels K: 3 (dog, cat and panda)

With above variables need a scoring function f that maps the images to the class label scores.
simle leniar scoring function: 
f(xi, W, b) = W.dot(xi) + b

each xi shape = (D, 1) [in animals dataset (3072, 1)]
W shape = (K, D) [in animals dataset case (3, 3072)]
b shape = (K, 1) [in animals dataset (3, 1)]
```  
Both the inputs xi and yi are fixed and not something we can modify 
(although we can obtain different xi's by applying different transformations 
to the input image) - but once we pass the image to a scoring function and 
loss function `the values donot change`.  
The only parameters that we have any control over 
* weight matrix `W`
* bias vector `b`  
The goal is to utilize the scoring and loss function to optimize 
the weight and bias vectors such that the classification accuracy increases.  
Exactly how we optimize the weight matrix depends on our loss function, 
but typically involves some form of gradient descent.  

#### Advantages of Parameterized Learning and Linear Classification
2 main advantages of using parameterized learning
1)  `Once we are done training our model we can discard the input data 
and keep only the weight matrix W and bias vector b`. This substantially reduces the size of our model 
since we need to store two sets of vectors (versus the entire training set).
2) `Classifying new test data is fast`. In order to perform classification all we need to do is to 
take dot product of W and xi, follow by adding in the bias b (i.e apply our scoring function).

