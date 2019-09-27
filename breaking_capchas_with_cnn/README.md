A deeep learning casestudy that will give an example of 
1. Downloading a set of images 
2. Labeling and annotating your images for training 
3. Training a CNN on the custom Dataset 
4. Evaluating and Testing the trained CNN 

The dataset of images that will be used is a set of captcha images used to prevent bots from 
automatically registering or logging into a given website (or, worse trying to brute force 
their way into someones account). 

### The Captcha breaker directory structure
|--- utilities 
|   |--- __init__.py 
|   |--- datasets 
|   |   |--- __init__.py 
|   |   |--- simpleedataseetloader.py 
|   |--- preprocessing 
|   |   |--- __init__.py 
|   |   |--- imagetoarraypreprocessor.py 
|   |   |--- simplepreprocessor.py 
|   |--- nn
|   |   |--- __init__.py
|   |   |--- conv
|   |   |   |--- __init__.py
|   |   |   |--- shallownet.py
|   |   |   |--- lenet.py
|   |--- callbacks
|   |   |--- __init__.py 
|   |   |--- trainingmonitor.py
|   |--- utils
|   |   |--- __init__.py
|   |   |--- captchahelper.py 
|--- captcha_breaker 
|   |--- dataset/
|   |--- downloads/
|   |--- output/
|   |--- annotate.py
|   |--- download_images.py
|   |--- test_model.py
|   |--- train_model.py

The captchahelper.py will store a utility function to help us process digits before feeding 
them into our deeep neural network. 

Also create a second directory called captcha_breaker outside utlities module.
The captcha breaker directory is where all our project code will be stored to break image captchas.
The dataset directory is where we will store our labeled digits which we will be hand-labeling. 
The datasets can be organised using the following directory structure.  
`root_directory/class_name/image_filename.jpg`  
our dataset directory will have the structure  
`dataset/{1-9}/example.jpg`  
where dataset is the root directory, {1-9} are the possible digit names, and example.jpg will 
be an example of the given digit. 
The downloads directory will store the raw captcha.jpg files downloaded from E-Zpass website. 
Inside the directory we will store our trained LeNet architecture.  

The download_images.py will be responsible for actually downloading the example captchas and 
saving them to disk. Once wee have downloaded a we'll need to extract the digits from each 
image and hand-labeled every digit

The train_model.py script will train LeNet on the labeled digits test_model.py will apply 
LeNet to captcha images themselves. 

