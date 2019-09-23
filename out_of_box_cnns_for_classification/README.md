Keras library ships with five CNN's that have been pre-trained on ImageNet dataset. 
* VGG16 
* VGG19
* ResNet50
* Inception V3
* Xception 

ImageNet Large Scale Visual Recognition Challenge (ILSVRC) is to train a model that can 
correctly classify an input image into 1000 separate object categories. These 1000 image 
categories represent object classes that we encounter in our day to day lifes such as species 
of dogs, cats, various household objects, vehicle types, and much more. 
This implies if we leverage CNNs pre-trained on the ImageNet dataset we can recognise all of 
these 1000 object categories out-of-box. The complete list of object categories that can be 
recognised using pre-trained ImageNet models can be found at 
(http://image-net.org/challenges/LSVRC/2014/browse-synsets). 

### State of the art CNNs in Keras 
As discussed previously in parameterised learning, it (pl) is two fold 
1. Define a machine learning model that can learn patterns from our input data during training 
time (requiring us to spend more time on training process), but have the testing process to be 
much faster. 
2. Obtain a model that can be deefined using a small number of parameters that can easily 
represent the network, regardless of training size. 

Hence the actual model is a function of its parameters, not the amount of training data. 
Training the model on 1 million images or 100 would result in the same output model size.  
Neural Networks front load major amount of work. Most of the time is spent on training the 
network (due to various factors eg: depth of architecture, amount of training data or number of 
experiments we have to run to to tune the hyper parameters). 
GPU helps in speeding up the training process as we need to perform both the forward pass 
and backward pass. The forward pass is much faster enabling us to classify input images using CPU

#### VGG16 and VGG19 
Image of VGG neetwork 

Introduced by Simonyan and Zisserman in their 2014 paper, Very Deep Convolutional Networks for 
Large Scale Image Recognition. 
VGG family of networks is characterized by using only 3x3 convolutional layers stacked on top 
of each other in increasing depth. Reducing volume size is handled by max pooling. 
Two fully connected layers each of 4,096 nodes are then followed by a softmax classifier. 

Two major drawbacks with VGG:
* It is painfully slow to train 
* The network weights themselves are quite large (in terms of disk space/bandwidth). 
Due to its depth and number of fully-connected nodes, the serialized weight files for 
VGG16 is 533MB while VGG19 is 574MB. 

#### ResNet
Image of ResNet 

Introduced by He et al. in their 2015 paper *Deep Residual Learning for Image recognition*, 
ResNet arch is seminal work in demonstrating that extremly deep networks can be trained using 
standard SGD (and a reasonable initialization function) through the use of residual modules. 
Further accuracy can be obtained by updating the residual modules to use identity mappings 
as demonstrated in their 2016 followup publication, Identity Mappings in Deep Residual Networks. 

ResNet50 has 50 weight layers and implementation in the Keras core library is based on former 
2015 paper. Even though ResNet is much deeper than both VGG16 and VGG19, the model size is 
substantially smaller due to the use of global average pooling rather than fully-connected layers, 
which reduces the model size down to 102MB for ResNet50. 

#### Inception V3
The Inception module (and the resulting Inception architecture) was introduced by Szegedy et al. 
their 2014 paper, *Going Deeper with Convolutions*. The goal of Inception module is to act as 
a "multi-level feature extractor" by computing 1x1, 3x3 and 5x5 convolutions within same module 
of the network - the output of these filters are then stacked along the channel dimension before 
being fed into the next layer in the network. 

The inception v3 included in the Keras core comes from the later by publication by Szegedy el al. 
*Rethinking the Inception Architecture for Computer Vision*. The weights for Inception V3 are 
smaller than both VGG and ResNet comming in at 96MB. 

#### Xception 
Proposed by Franchios Chollet in his 2016 paper, *Xception: Deeep Learning with Depthwise Separable Convolutions*. 
It is an extension to the Inception architecture which replaces the standard Inception modules with 
depth wise separable convolutions. The Xception's weights are the smallest of the pre-trained 
convolutions included in the Keras library, weighing only 91MB. 

#### Can we go smaller
SqueezeNet architecture is often used when we need a tiny footprint (It is not included in 
keras's core models). SqueezeNet is very small 4.9MB often used when networks need to be trained 
and deployed over a network and/or to resource constrained devices. 

