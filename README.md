# Grain1
Simple classification ML algorithm to sort grain

## First we got the data from grainspace: https://github.com/hellodfan/GrainSpace
we will focus mainly in the M600 data, which is collected with smartphones cameras, focusing on simplicity
then i'll switch to G600 which is collected with industrial cameras, focusing  precision.

So far there's a preprocess algorithm, with resizes and normalizes images of maize.
it can preprocess wheat as well.

The first models have been created, for maize and wheat separately.

There still need to stratify the classes of abnormal states of the grain and train a model that can classify both grains at the same time.
Also there is need to optimize the amount of layers of the neural network and being able to monitorize the resulting models in tensorboard.
