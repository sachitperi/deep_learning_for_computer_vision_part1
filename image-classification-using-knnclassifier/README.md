### Project Structure
```
|--utilities  
|	|--__init__.py  
|	|--datasets  
|	|	|--__init__.py  
|	|	|--simpledatasetloader.py  
|	|--preprocessing  
|	|	|--__init__.py  
|	|	|--simplepreprocessor.py  
```

#### Dataset Submodule
* Implementation of a class named SimpleDatasetLoader 
* We will be using this class to load small image datasets into disk (that can fit into memory), optionally preprocess each image in the dataset according to a set of functions and then return the 
1) Images (raw pixel intensities)
2) Class label associated with each image

#### Preprocessing Submodule 
There are a number of preprocessing methods that we can apply to our datasets of images top boost classification accuracy, including mean subtraction, sampling random patches, or simply resizing an image to a fixed size. 
Currently our SimplePreproceessor class will - Load an image from disk and resize it to a fixed size, ignoring aspect ratio. 

#### Basic Image Preprocessor 
* ML algos such as k-NN, SVM's, and even Convolutional Neural Networks require all images in the dataset to have a fixed feature vector size. 
* In case of images this implies that our images must be preprocessed and scaled to have identical widths and height

There are multiple ways to accomplish this resizing and scaling, 
* Respect the aspect ratio of the original image to the scaled image (More advanced method)
* Ignore the aspect ratio and simply squash the width and height to the required dimensions

Exactly which method to use depends on the complexity of the problem's `factors of variations`, in some cases ignoring the aspect ratio works fine in other cases preserving the aspect ratio 
For exploring the k-NN classifier we will start with the basic solution: building an image preprocessor that resizes the image, ignoring the aspect ratio. The code can be found in `simpreprocessor.py`.
      
#### Pros and Cons of k-NN 
Pros:
* verry simple to impliment and to understand 
* the classifier takes about no time since all we need to do is to store our datapoints for the purpose of later computing distance to them and obtaining our final classification.

Cons: 
* Classifying a new test point requires a comparision to every single datapoint in our training data,  which scales O(N), making work with larger datasets computationally prohibitive
* k-NN algo is more suited for low dimensional feature space (which images are not)
* k-NN algorithm dosent learn anything. The algorithm is not able to make itself smarter if it makes mistakes; its simply relying on distancces in n-dimensional space to make the classification 
 