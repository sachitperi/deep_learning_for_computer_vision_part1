### Intro

Traditional feedforward neural network 
every neuron in the input layer is connected to every output neuron in the output
layer. In traditional NN's fully connected (FC) layers are used. 
In CNN's the FC layer is not used until the very last. CNN is a NN which swaps 
in a specialized "convolutional" layer in place of "fully connected" layer 
for atleast one of the network.
A nonlinear activation such as Relu is applied to the output of the convolutions 
and the proceess convolution => activation continues 
(along with a mixture of other layer types to help reduce the width, height 
of input volume and help reduce overfitting) until we reach the end of the network 
and apply the FC layers where we obtain the final output classifications. 

In context of image classification a CNN may learn to:
* Detect the edges from a raw pixel data in the first layer.
* Use these edges to detect shapes (i.e blobs) in the second layer
* Use these shapes to detect higher-level features such as facial structures, 
parts of a car  

CNN gives us the two benifits
* Local Invariance: Allows us to classify an image as containing a particular 
object regardless of where in the image the object appears. 
We obtain this local invariance through the usage of "pooling layers" which 
identifies reegions of our input volume with a high response ton a particular filter.
* Compositionality:Each filter composes a local patch of lower-level features 
into a higher-level repreesentation, similar to how we can compose a set of 
mathematical functions that build on the output of previous functions: 
f(g(x(h(x)))) - this composition allows our network to learn more rich features 
deeper in the network.


#### Understanding convolutions
* What are image convolutions? 
* What do they do? 
* Why do we use them? 
* How do we apply them to images? 
* What role do convolutions play in deep learning? 

Neural Networks operating on raw pixel intensities 
* do not scale well as the image size increases
* Leaves much accuracy to be desired (i.e, a standard feed forward 
enroll network on CIFAR10 obtained only 15% accuracy)

eg:- Eacch image in CIFAR10 dataset is of shape 32 x 32 x 3 = 3072. 
3072 inputs doesnt seem much but for the same .   

##### Layer Types
There are many layers used to build convolutional neural networks. 
The most common ones are as follows
* Convolutional (CONV)
* Activation (ACT or RELU where we use the same of the actual activation function)
* Pooling (POOL)
* Fully Connected (FC)
* Batch Normalization (BN)
* Dropout (DO)

Of these layer types Convolution `CONV, FC (and to a lesser extent BN)` 
are the only ones that contains parameters to be learned during the 
training process.

##### Convolutional Layers 
The CONV layer consists of a set of k learnable features (i.e kernels), where 
each filter consists of a width and a height, and are nearly always square.
These features are small but extend throughout the full depth of the volume.

For inputs to a CNN the depth is the number of channels in the image 
(i.e depth of 3 when working with RGB images, one for each channel). 
For volumes deeper the **depth will be number of filters applied in the previous layer**

The concept of convolving a small filter with a larger input volume has a 
special meaning in Convolutional Neural Networks - specifically, the 
`local connectivity` and `reeceptive field` of a neuron. When working with 
images its always impractical to connect neurons in current volume to all neurons 
there are simply too many connections and too many weights making it impossible 
to train deep networks on images with large spatial dimensions. Instead, when 
utilizing CNN's we choose to connect each neuron to only a local region 
the `receptive field` (or simply the variable F) of the neuron.
The **receptive field `F`** is the size of the filter, yielding an FxF kernel 
with the input volume.

There are three parameters that control the output volume: the **depth**, 
**stride** and **zero-padding** size.

###### Depth 
The depth of the output volume controls the number of neurons (i.e filters) 
in the CONV layers that connect to the local region of the input volume. 
Each filter produces an activation map that `activate` in the presence of 
oriented edges or blobs of color. 
For a given CONV layer, the depth of activation map will be K, or simply the 
number of filters we are learning in current layer. The set of filters that are 
"looking at" the same (x, y) location of input called the **depth column**.

###### Stride 
"sliding" a small matrix across a larger matrix, stopping at each coordinate, 
computing an element wise multiplication and sum, then storing the output. 
When creating CONV layers we generally use a stride of S=1 or S=2.

###### Zeero-Padding 
We need to pad the borders of an image to retain the original 
image image size when applying a convolution. 
The amount of padding we apply is controlled by the parameter P. 

We can compute the size of output volume as a function of the 
input volume size (W, assuming the input images are square) the receptive field size F, 
the Stride S, and the amount of padding P

((W-F+2P)/S)+1

If the above equation dosent output an integer then the strides are set incorrectly. 

###### Summary 
* Accepts an input volume of size W<sub>input</sub> x H<sub>input</sub> x D<sub>input</sub> 
(The inputs aree normally square, so its common to see W<sub>input</sub> = H<sub>input</sub>)
* Requires Four parameters 
1. Thee number of filters K (Which controls the depth of the volume)
2. The receeptive field size F (The size of the K kernels used for 
convolution and is nearly always square)
3. The stride S
4. The amount of zero-padding P.
* The output of the CONV layer is then 
W<sub>output</sub> x H<sub>output</sub> D<sub>output</sub> where
    * W<sub>output</sub> = ((W<sub>input</sub> - F + 2P)/S) + 1
    * H<sub>output</sub> = ((H<sub>input</sub> - F + 2P)/S) + 1
    * D<sub>output</sub> = K 
    
##### Activation Layer
After each CONV layer in CNN we apply a nonlinear activation function, such as 
ReLU, ELU or any of the other Leaky ReLU variants. 
An activation function accepts an input volume of size 
W<sub>input</sub> x H<sub>input</sub> x D<sub>input</sub> then applies the 
given activation function. Since the activation function is applied in an 
element-wise manner, the output of an activation layer is always 
the same as the input dimension, W<sub>output</sub> = W<sub>input</sub>, 
H<sub>output</sub> = H<sub>input</sub>, 
D<sub>output</sub> = D<sub>input</sub>

##### Pooling Layer 
There are two ways to reduce the size of an input volume 
1. CONV layers with a stride > 1
2. Pool layers

Pool layers operate on each of the depth slices of an input independently 
using either the max or average function. Max pooling is typically done in 
the middle of a CNN architecture to reduce spatial size , whereas average 
pooling is normally used as final layer of network 
(GoogLeNet, SequenceNet, ResNet) where we wish to avoid FC layers entirely 

We can further decrease the size of our own output by increasing the strides. 
If we use S=2, For every 2x2 block in the output 
we keep only the largest value, then take a step of 2 pixels, and apply the 
operation again. This pooling allows us to reduce the size by a factor of 2, 
effectively discarding 75% of the activations from the previous layer. 

Pooling layers accept an input volume of size 
W<sub>input</sub> x H<sub>input</sub> x D<sub>input</sub>. 
They then require 2 parameters 
* Receptive field size F
* The stride S  
Applying pool operations yields an output volume of size 
W<sub>output</sub> x H<sub>output</sub> x D<sub>output</sub>  
* W<sub>output</sub> = ((W<sub>input</sub> - F)/S) + 1
* H<sub>output</sub> = ((H<sub>input</sub> - F)/S) + 1
* D<sub>output</sub> = D<sub>input</sub> 

###### To Pool or to Conv?
In 2014 paper in, *Striving for Simplicity: The All Convolutional Net* 
it is recomended to discard the pooling layers and rely on 
CONV layers with larger strides to handel down sampeling the spatial 
dimensions of volume. The work demonstrated this approach works 
very well on a variety of datasets, Including the CIFAR10 
(small images, low number of classes) and imagenet 
(Large input images, 1000 classes). This trend continues with 
ResNet architecture which uses CONV layers for down sampling as well.  

###### Fully Connected Layers

##### Batch Normalization
Used to normalize the activations of a given input volumes before 
passing it on to the next layer in the network. 

##### Dropout
Dropout is a form of regularization that aims to help prevent overfitting by 
increasing testing accuracy. 

#### Rules of thumb
Common rules when constructing own CNN's 
* Images represented in input layers should be square. 
(Helps us to use the linear algebra optimizations libraries) 
Common input layers include 32x32, 64x64, 96x96, 224x224, 227x227, and 229x229 
* The input layer should be divisible by 2 multiple times after the first conv 
layer is applied (can be done by tweaking the filter size and stride).
TThe "divisible by two" eenablees the spatial inputs in our network to be 
conveniently down sampled via pool operation in an efficient manner.
* In generaal CONV layers should use small filter sizes like 3x3, 5x5. 
Tiny 1x1 filters are used to learn local features that too only in large networks.
Large filters like 7x7, 11x11 are used as the first CONV layer in the network 
to reduce spatial input size, provided the images are sufficiently larger > 200x200 pixels.  
* Commonly use a stride of S=1 for CONV layers to learn filters while the 
POOL layer is responsible for down sampling.