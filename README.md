 
---

<div align="center">    
 
# Behavior Cloning :arrows_clockwise: with Rovers :red_car:

## Purpose & Objectives
We’re going to attempt to teach a robotic rover how to use a line as a guide for driving around a race track.  This problem will be solved using a single sensor – a simple RGB camera.  All of the control will be derived from a single image, a single frame from streamed video that represents a snapshot in time. This poses both a challenge and a great opportunity for applied machine learning.  As discussed, we will approach this problem by applying a technique called behavioral cloning – a subset under a larger group of techniques called imitation learning (IL). 

In general, at the heart of IL is a Markov Decision Process (MDP) where we have a set of states S, a set of actions A, and a P(s’|s,a) transition model (the probability that an action a in the state s leads to s’).  This may or may not be associated with an unknown reward function, R(s,a).  For now, there will be no R function.  If you’re thinking to yourself that this more or less resembles reinforcement learning (RL), then you’d be correct; however, our rovers will attempt to learn an expert’s optimal policy π* through a simple supervised learning approach.

## Description of the Problem
The environment consists of the following: (1) State s ' S is represented by an observation through our video camera (i.e. a frame/image), along with any other data you might find useful.  Our camera has a resolution of 640x480 with 3 8-bit (RGB) color channels.  (2) A deceptively simple action space, that allows for a throttle amount and a steering amount (remember, this is a skid steer system).  The goal is to perform n laps around a track defined by colored material (some sort of matte tape) laid out in an arbitrary line pattern in the fastest time possible. Ideally, our model will generalize to drive around any shape of track.

## Artifacts
### 2 - Models :link:
### 3 - Data Repo :link:
### 4 - Experiment Logs :link:
4.a Data Processing <br>
4.b Data Collection <br>
4.c Model Creation <br>
    
### 5 - Contributers and Acknowledements







# baseline cnn model for mnist
from numpy import mean
from numpy import std
from matplotlib import pyplot
from sklearn.model_selection import KFold
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D
from tensorflow.keras.utils import to_categorical
from keras.layers import Dense
from keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import layers
from keras.optimizers import adam_v2

# load train and test dataset
from tensorflow.python.client.device_lib import list_local_devices


def load_dataset():
    # load dataset
    (trainX, trainY), (testX, testY) = mnist.load_data()
    # reshape dataset to have a single channel
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    # one hot encode target values
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY


# scale pixels
def prep_pixels(train, test):
    # convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # return normalized images
    return train_norm, test_norm


def define_model():
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Flatten()(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.Dense(10, activation="softmax")(x)
    opt = SGD(lr=0.1, momentum=0.9)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss="categorical_crossentropy",
                  optimizer=opt,
                  metrics=["accuracy"])
    print(model.summary())
    return model


# evaluate a model using k-fold cross-validation
def evaluate_model(dataX, dataY, n_folds=3):
    scores, histories = list(), list()
    # prepare cross validation
    kfold = KFold(n_folds, shuffle=True, random_state=1)
    # enumerate splits
    for train_ix, test_ix in kfold.split(dataX):
        # define model
        model = define_model()
        # select rows for train and test
        trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
        # fit model
        with tf.device('/GPU:0'):
            history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=1)
            # evaluate model
            _, acc = model.evaluate(testX, testY, verbose=1)
            print('> %.3f' % (acc * 100.0))
            # stores scores
            scores.append(acc)
            histories.append(history)
    return scores, histories


# plot diagnostic learning curves
def summarize_diagnostics(histories):
    for i in range(len(histories)):
        # plot loss
        pyplot.subplot(2, 1, 1)
        pyplot.title('Cross Entropy Loss')
        pyplot.plot(histories[i].history['loss'], color='blue', label='train')
        pyplot.plot(histories[i].history['val_loss'], color='orange', label='test')
        # plot accuracy
        pyplot.subplot(2, 1, 2)
        pyplot.title('Classification Accuracy')
        pyplot.plot(histories[i].history['accuracy'], color='blue', label='train')
        pyplot.plot(histories[i].history['val_accuracy'], color='orange', label='test')
    pyplot.show()


# summarize model performance
def summarize_performance(scores):
    # print summary
    print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores) * 100, std(scores) * 100, len(scores)))
    # box and whisker plots of results
    pyplot.boxplot(scores)
    pyplot.show()


# run the test harness for evaluating a model
def run_test_harness():
    # load dataset
    trainX, trainY, testX, testY = load_dataset()
    # prepare pixel data
    trainX, testX = prep_pixels(trainX, testX)
    # evaluate model
    scores, histories = evaluate_model(trainX, trainY)
    # learning curves
    summarize_diagnostics(histories)
    # summarize estimated performance
    summarize_performance(scores)


# entry point, run the test harness

import tensorflow as tf

if __name__ == '__main__':
    run_test_harness()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
