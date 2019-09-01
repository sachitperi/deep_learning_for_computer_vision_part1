VGGNet was introduced by Simonyan and Zisserman in their 2014 paper, 
Very Deep Learning Convolutional Neural Networks for Large Scale Image Recognition. 
The primary contribution of their work was demonstrating that an architecture with 
very small (3x3) filters can be trained to increasingly higher depths (16-19 Layers) 
and obtain state-of-the-art classification on the challenging ImageNet classification 
challenge.  
Previously, network architecture in deep learning literature used a mix of filter sizes: 
the first layer of CNN usually includes filter sizes somewhere between 7x7 and 11x11. 
from there filter sizes progressively reduced to 5x5. Finally only the deepest layers 
of network used 3x3 filters.  
VGGNet is unique in that it uses 3x3 filter throughout the entire architecture. 
The use of these small kernels is arguably what helps VGGNet generalize to 
classification problems outside what the network was originally trained on. 

Objective is to review the VGG family of networks and define what 
characteristics a CNN must exhibit to fit into this family. From there we will 
impliment a smaller version of VGGNet called miniVGGNet that can easily be trained 
in a low power local machine. This will also demonstrate how to use the two 
important layers discussed in `cnn` - batch normalization (BN) and Dropout 

## VGG Family of Networks 
The VGG family of CNN's can be characterized by two key components: 
1) All conv layers in the network using only 3x3 filters 
2) Stacking multiple CONV => RELU layer sets 
(where the number of consecutive CONV => RELU layers normally increases the deeper 
we go in the network)

### The (Mini) VGG Architecture 
In both ShallowNet and LeNet a series of CONV => RELU => POOL layers were applied. 
However in VGGNet, we stack multiple CONV => RELU layers prior to applying a single 
pool layer. Doing so allows the network to learn more rich features from the CONV 
layers prior to the down sampling the spatial input size via the POOL operation. 

Overall miniVGGNet consists of 2 sets of CONV => RELU => CONV => RELU => POOL layers 
followed by a set of FC => RELU => FC => SOFTMAX layers. The first two CONV layers 
will learn 32 filters of size 3x3. The second two CONV layers will learn 64 filters, 
again, each of size 3x3. Our pool layers will perform max pooling over a 2x2 window 
with a 2x2 stride. We will also be implementing batch normalization layers after 
activations along with dropout layers (DO) after the POOL and FC layers. 

Network Architecture

| Layer Type     | Output Size  | Filter Size/Stride |
| :------------- | :----------: | ----------------:  |
| INPUT IMAGE    | 32×32×3      |                    |
| CONV           | 32×32×32     | 3x3, k=32          |
| ACT            | 32×32×32     |                    |
| BN             | 32×32×32     | 2x2                |
| CONV           | 32×32×32     | 3x3, k=32          |
| ACT            | 32×32×32     |                    |
| BN             | 32×32×32     |                    |
| POOL           | 16x16x32     | 2x2                |
| DROPOUT        | 16×16×32     |                    |
| CONV           | 16×16×64     | 3x3, k=64          |
| ACT            | 16×16×64     |                    |
| BN             | 16×16×64     | 2x2                |
| CONV           | 16×16×64     | 3x3, k=32          |
| ACT            | 16×16×64     |                    |
| BN             | 16×16×64     |                    |
| POOL           | 8x8x64       | 2x2                |
| DROPOUT        | 8×8×64       |                    |
| FC             | 512          |                    |
| ACT            | 512          |                    |
| BN             | 512          |                    |
| DROPOUT        | 512          |                    |
| FC             | 10           |                    |
| SOFTMAX        | 10           |                    |

### Implementing MiniVGGNet 
```project Structuree
|--- lenet_mnist.py
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
|   |   |   |--- minivggnet.py 
```

### MiniVGGNet on CIFAR10 
* Load the dataset from disk 
* Instantiate the MiniVGGNet architecture 
* Train the MiniVGGNet using the training data 
* Evaluate network performance using testing data 
