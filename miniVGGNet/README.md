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

``` 
Train on 50000 samples, validate on 10000 samples
Epoch 1/40
50000/50000 [==============================] - 277s 6ms/step - loss: 3.3761 - acc: 0.2815 - val_loss: 2.4108 - val_acc: 0.1918
Epoch 2/40
50000/50000 [==============================] - 349s 7ms/step - loss: 2.0031 - acc: 0.3350 - val_loss: 1.9332 - val_acc: 0.3491
Epoch 3/40
50000/50000 [==============================] - 379s 8ms/step - loss: 1.7605 - acc: 0.3747 - val_loss: 1.4549 - val_acc: 0.4746
Epoch 4/40
50000/50000 [==============================] - 341s 7ms/step - loss: 1.6582 - acc: 0.4043 - val_loss: 1.4060 - val_acc: 0.4976
Epoch 5/40
50000/50000 [==============================] - 327s 7ms/step - loss: 1.6906 - acc: 0.4031 - val_loss: 2.5518 - val_acc: 0.2814
Epoch 6/40
50000/50000 [==============================] - 335s 7ms/step - loss: 1.7717 - acc: 0.3934 - val_loss: 7.6569 - val_acc: 0.1342
Epoch 7/40
50000/50000 [==============================] - 326s 7ms/step - loss: 1.7787 - acc: 0.3865 - val_loss: 1.4425 - val_acc: 0.4677
Epoch 8/40
50000/50000 [==============================] - 349s 7ms/step - loss: 1.6037 - acc: 0.4234 - val_loss: 1.3803 - val_acc: 0.5007
Epoch 9/40
50000/50000 [==============================] - 342s 7ms/step - loss: 1.5576 - acc: 0.4409 - val_loss: 1.3367 - val_acc: 0.5170
Epoch 10/40
50000/50000 [==============================] - 342s 7ms/step - loss: 1.5230 - acc: 0.4524 - val_loss: 1.3142 - val_acc: 0.5226
Epoch 11/40
50000/50000 [==============================] - 371s 7ms/step - loss: 1.4887 - acc: 0.4670 - val_loss: 1.3018 - val_acc: 0.5397
Epoch 12/40
50000/50000 [==============================] - 313s 6ms/step - loss: 1.4710 - acc: 0.4717 - val_loss: 1.2579 - val_acc: 0.5429
Epoch 13/40
50000/50000 [==============================] - 303s 6ms/step - loss: 1.4458 - acc: 0.4785 - val_loss: 1.2715 - val_acc: 0.5394
Epoch 14/40
50000/50000 [==============================] - 288s 6ms/step - loss: 1.4285 - acc: 0.4870 - val_loss: 1.3272 - val_acc: 0.5233
Epoch 15/40
50000/50000 [==============================] - 296s 6ms/step - loss: 1.4153 - acc: 0.4876 - val_loss: 1.2220 - val_acc: 0.5642
Epoch 16/40
50000/50000 [==============================] - 313s 6ms/step - loss: 1.3962 - acc: 0.4973 - val_loss: 1.2149 - val_acc: 0.5643
Epoch 17/40
50000/50000 [==============================] - 286s 6ms/step - loss: 1.3826 - acc: 0.5034 - val_loss: 1.1987 - val_acc: 0.5732
Epoch 18/40
50000/50000 [==============================] - 290s 6ms/step - loss: 1.3680 - acc: 0.5087 - val_loss: 1.1846 - val_acc: 0.5803
Epoch 19/40
50000/50000 [==============================] - 289s 6ms/step - loss: 1.3489 - acc: 0.5144 - val_loss: 1.1646 - val_acc: 0.5819
Epoch 20/40
50000/50000 [==============================] - 291s 6ms/step - loss: 1.3421 - acc: 0.5188 - val_loss: 1.1516 - val_acc: 0.5937
Epoch 21/40
50000/50000 [==============================] - 292s 6ms/step - loss: 1.3253 - acc: 0.5246 - val_loss: 1.1550 - val_acc: 0.5829
Epoch 22/40
50000/50000 [==============================] - 292s 6ms/step - loss: 1.3178 - acc: 0.5283 - val_loss: 1.1265 - val_acc: 0.5991
Epoch 23/40
50000/50000 [==============================] - 291s 6ms/step - loss: 1.3030 - acc: 0.5325 - val_loss: 1.1323 - val_acc: 0.5941
Epoch 24/40
50000/50000 [==============================] - 291s 6ms/step - loss: 1.2958 - acc: 0.5360 - val_loss: 1.1400 - val_acc: 0.5869
Epoch 25/40
50000/50000 [==============================] - 289s 6ms/step - loss: 1.2855 - acc: 0.5377 - val_loss: 1.1927 - val_acc: 0.5661
Epoch 26/40
50000/50000 [==============================] - 285s 6ms/step - loss: 1.2786 - acc: 0.5405 - val_loss: 1.1062 - val_acc: 0.6092
Epoch 27/40
50000/50000 [==============================] - 288s 6ms/step - loss: 1.2714 - acc: 0.5431 - val_loss: 1.1115 - val_acc: 0.6069
Epoch 28/40
50000/50000 [==============================] - 292s 6ms/step - loss: 1.2616 - acc: 0.5503 - val_loss: 1.0809 - val_acc: 0.6192
Epoch 29/40
50000/50000 [==============================] - 291s 6ms/step - loss: 1.2537 - acc: 0.5502 - val_loss: 1.1071 - val_acc: 0.6044
Epoch 30/40
50000/50000 [==============================] - 281s 6ms/step - loss: 1.2500 - acc: 0.5514 - val_loss: 1.0741 - val_acc: 0.6208
Epoch 31/40
50000/50000 [==============================] - 281s 6ms/step - loss: 1.2413 - acc: 0.5574 - val_loss: 1.0640 - val_acc: 0.6213
Epoch 32/40
50000/50000 [==============================] - 277s 6ms/step - loss: 1.2400 - acc: 0.5583 - val_loss: 1.0643 - val_acc: 0.6257
Epoch 33/40
50000/50000 [==============================] - 287s 6ms/step - loss: 1.2300 - acc: 0.5594 - val_loss: 1.0708 - val_acc: 0.6183
Epoch 34/40
50000/50000 [==============================] - 288s 6ms/step - loss: 1.2305 - acc: 0.5602 - val_loss: 1.0884 - val_acc: 0.6211
Epoch 35/40
50000/50000 [==============================] - 292s 6ms/step - loss: 1.2206 - acc: 0.5642 - val_loss: 1.0703 - val_acc: 0.6196
Epoch 36/40
50000/50000 [==============================] - 300s 6ms/step - loss: 1.2189 - acc: 0.5647 - val_loss: 1.0884 - val_acc: 0.6212
Epoch 37/40
50000/50000 [==============================] - 306s 6ms/step - loss: 1.2105 - acc: 0.5691 - val_loss: 1.0670 - val_acc: 0.6242
Epoch 38/40
50000/50000 [==============================] - 296s 6ms/step - loss: 1.2104 - acc: 0.5666 - val_loss: 1.0464 - val_acc: 0.6344
Epoch 39/40
50000/50000 [==============================] - 289s 6ms/step - loss: 1.2173 - acc: 0.5652 - val_loss: 1.1158 - val_acc: 0.6071
Epoch 40/40
50000/50000 [==============================] - 302s 6ms/step - loss: 1.2115 - acc: 0.5660 - val_loss: 1.0582 - val_acc: 0.6258
[INFO] evaluating the network...
              precision    recall  f1-score   support

    airplane       0.61      0.67      0.64      1000
  automobile       0.70      0.84      0.77      1000
        bird       0.45      0.53      0.48      1000
         cat       0.45      0.43      0.44      1000
        deer       0.62      0.49      0.55      1000
         dog       0.56      0.50      0.53      1000
        frog       0.73      0.70      0.72      1000
       horse       0.72      0.68      0.70      1000
        ship       0.66      0.78      0.72      1000
       truck       0.80      0.62      0.70      1000

   micro avg       0.63      0.63      0.63     10000
   macro avg       0.63      0.63      0.62     10000
weighted avg       0.63      0.63      0.62     10000
```