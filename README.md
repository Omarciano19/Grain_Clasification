# Grain_ML

The **objective** of this project is develop neural networks each time more efficient to clasify wheat and maize grain as healty, or in one of the six unacceptable categories.

## Data collection:
First we got the data from [grainspace](https://github.com/hellodfan/GrainSpace) 

we will focus first in the M600 data, which is collected with smartphones cameras, focusing on simplicity
then i'll switch to G600 which is collected with industrial cameras, focusing on precision.

## Progress:
So far there are three notebooks presenting the next progress.

### A preprocess notebook
it resizes and normalizes images of maize and wheat under the M600 or G600 grainspace folder, then it divides the images in training and test sets in a 70:30 proportion and and orders the images in the structure needed by keras.

### Grain1.2
Simple classification ML neural network to sort grain with four convolutional layers and one dense layer;
```
 dense_layers = 1
layer_sizes = 32
conv_layers = 4
```
it takes the images of wheat and maize in low resolution (M600 grainspace dataset) separatly to create two diferent models that clasify only one kind of grain at the time in one of the seven posible conditions of each grain. 

The training is abalible using a *ImageDataGenerator* wich feds *keras* with data of the images optimized for training, being croped, rotated, fliped, and shuffled.

The second version of this notebook is able to train the neural network using a GPU.
TensorBoard is avalible through a log_dir with histograms of accuaracy and loss in an absolute of "per epoch" way.

### Grain 2-2
A notebook that creates many  models through the iteration of the next parameters and values:

```
 dense_layers = [0, 1, 2]
layer_sizes = [32, 64, 128]
conv_layers = [1, 2, 3] 
```
each one of them can clasify wheat and maize at the same time as healty or in any of the six possible conditions for each kind of grain with different accuaracies. With the integration of TensorFlow-GPU and TensorBoard the notebook outputs many logs that needs to be analized in order to find the one with a lower loss funtion and greater accuaracy.

### GrainG600
Similar to *Grain 2-2*, but this notebook uses the high resolution dataset (Grainspace G600), with better data the results improve significally.

### GrainG600_Keras_tuner

Notebook wich also uses the high resolution data (Grainspace G600) and creates a single model to clasify both kind of grain in one of the 14 possible categories.

It takles the creation of models with the keras tuner,  a  funtion *model* is defined wich takes  the hiperparameters of the model as parameters and  creates a model, then we fed the function to a keras tuner wich use the "Hyperband" method to search for the best hyperparameters to optimize the selected metric, *loss*. The tuner also generate tensorboard logs to monitor the training process.

Then, we obtain the best hiperparameters and train the model using the funtion.


### GrainG600_tunning (name change required)

This notebook have the same objectives as *GrainG600_Keras_tuner*  but embraces the use of *keras tuner* in a optimal way, in this notebook we define the hiperparameter space and it grid, 
```
HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([16, 32]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.2))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))
```
also the metrics of interest
```
METRIC_ACCURACY = 'accuracy'
METRIC_LOSS = 'loss'
```
the *train_test_model* function wich takes the hiperparameters as input and defines a model architecture, compiles it and train it to return the loss and accuaracy of a model; also the function *run* is defined, it runs *train_test_model" and records the values of accuaracy and loss.

Finally this funtion is used in three for loops wich explore the space of hiperparameters.







