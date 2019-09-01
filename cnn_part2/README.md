# Goal
* Implement few CNN's using python and keras.  
    * quick review of keras configurations to keep in mind when training CNN's.
* Implement ShalowNet (a very shallow network with only a single CONV layer).

## Keras Configurations and Converting Images to Arrays 
Review keras.json configuration file and how the settings inside this file 
will influence how you implement your own CNN. 
Also implement a second image processor named ImageToArrayPreprocessor which accepts 
an input image and converts it to a numpy array that Keras can work with. 

### Understanding the keras.json file
 
``` Default keras.json in ~/.keras/keras.json
{
    "floatx": "float32",
    "epsilon": 1e-07,
    "backend": "tensorflow",
    "image_data_format": "channels_last"
}

```
1. `epsilon` value is used in a variety of locations throughout the 
Keras Library. The default value `1e-07` is suitable and should not be changed. 
2. `floatx` defines the floating point precision - it is safe to leave this 
value to float32. 
3. `backend` By default, the Keras library uses the TensorFlow numerical 
computation backend.  We can also use the Theano backend simply by replacing tensorflow with theano. 
4. image_data_format which can accept two values: channels_last or channels_first. 

#### Image to Array Preprocessor
Keras library allows an image_to_array function that accepts an input image and 
then properly orders the channels based on image_data_format. 

```project structure
|--- utilities 
|   |--- __init__.py 
|   |--- datasets 
|   |   |--- __init__.py 
|   |   |--- simpleedataseetloader.py 
|   |--- preprocessing 
|   |   |--- __init__.py 
|   |   |--- imagetoarraypreprocessor.py 
|   |   |--- simplepreprocessor.py 
```

## ShallowNet 
ShallowNet contains only a few layers. The entire network can be summarized as 
INPUT => CONV => RELU => FC 

### Implementing ShallowNet 
```project structure
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
```

### ShallowNet on Animals dataset
```Results when running the shallowNet with lr=0.0009
Epoch 100/100
2250/2250 [==============================] - 1s 292us/step - loss: 0.7081 - acc: 0.6778 - val_loss: 0.8027 - val_acc: 0.5880
[INFO] evaluating the network...
              precision    recall  f1-score   support

        cats       0.51      0.57      0.54       249
        dogs       0.48      0.49      0.49       262
       panda       0.81      0.72      0.76       239

   micro avg       0.59      0.59      0.59       750
   macro avg       0.60      0.59      0.60       750
weighted avg       0.60      0.59      0.59       750

```

### ShallowNet on CIFAR10
```Results when running the shallowNet with lr=0.003 
Epoch 100/100 
Epoch 40/40 
50000/50000 [==============================] - 17s 333us/step - loss: 0.9917 - acc: 0.6590 - val_loss: 1.1950 - val_acc: 0.5811 
[INFO] evaluating the network... 
              precision    recall  f1-score   support

    airplane       0.66      0.58      0.62      1000
  automobile       0.73      0.67      0.70      1000
        bird       0.52      0.35      0.42      1000
         cat       0.39      0.48      0.43      1000
        deer       0.44      0.63      0.52      1000
         dog       0.57      0.36      0.45      1000
        frog       0.64      0.66      0.65      1000
       horse       0.70      0.61      0.65      1000
        ship       0.61      0.79      0.69      1000
       truck       0.64      0.68      0.66      1000

   micro avg       0.58      0.58      0.58     10000
   macro avg       0.59      0.58      0.58     10000
weighted avg       0.59      0.58      0.58     10000

```

