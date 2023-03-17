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


# callbacks
from tf.keras.callbacks import LearningRateScheduler 
from tf.keras.callbacks import EarlyStopping
# checkpointing
CHECK_POINT_FILEPATH = ''

from tensorflow.python.client.device_lib import list_local_devices


def load_dataset():
    pass



def define_model():
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = layers.Conv2D(filters=64,
                    kernel_size=3,
                    activation="relu",
                    kernel_initializer="truncated_normal")(inputs)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Dropout(.5)(x)


    x = layers.Conv2D(filters=128,
                     kernel_size=3, 
                     activation="relu",
                     kernel_initializer="truncated_normal")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Dropout(.5)(x)


    x = layers.Conv2D(filters=256,
                     kernel_size=3, 
                     activation="relu",
                     kernel_initializer="truncated_normal")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Dropout(.5)(x)


    x = layers.Flatten()(x)
    x = layers.BatchNormalization()(x)

    outputs = layers.Dense(2)(x)
    opt = SGD(lr=0.1, momentum=0.9)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss="mse", 
                  optimizer=adam_v2.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, clipvalue=0.1),
                  metrics=["accuracy", "mae"]
    print(model.summary())
    return model


#------ Callbacks ------
model_early_stoping_callback = EarlyStopping(monitor='loss', patience=3)

model_checkpoint_callback = ModelCheckpoint(
    filepath=CHECK_POINT_FILEPATH,
    save_weights_only=False,
    monitor='val_acc',
    mode='max',
    save_best_only=True)

def scheduler(epoch):
  if epoch < 10:
    return 0.001
  else:
    return 0.001 * tf.math.exp(0.1 * (10 - epoch))

model_learning_rate_scheduler_callback = LearningRateScheduler(scheduler)

callbacks = [model_early_stoping_callback,
            model_checkpoint_callback,
            model_learning_rate_scheduler_callback]


#------ Callbacks ------

# evaluate a model using k-fold cross-validation
def train(dataX, dataY,):
    scores, histories = list(), list()
    # fit model
    with tf.device('/GPU:0'):
        history = model.fit(trainX, trainY, 
                            epochs=10, batch_size=32, 
                            validation_data=(testX, testY),
                            callbacks = callbacks,
                            verbose=1)
        # evaluate model
        _, acc = model.evaluate(testX, testY, verbose=1)
        print('> %.3f' % (acc * 100.0))
        # stores scores
        scores.append(acc)
        histories.append(history)
return scores, histories


# plot diagnostic learning curves
def plot_loss(histories):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def plot_acc(histories):
    plt.clf()                          
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    plt.plot(epochs, acc, "bo", label="Training accuracy")
    plt.plot(epochs, val_acc, "b", label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


# run the test harness for evaluating a model
def train_harness():
    # load dataset
    trainX, trainY, testX, testY = load_dataset()
    # evaluate model
    scores, histories = train(trainX, trainY)
    
    # learning curves
    plot_acc(histories)
    plot_loss(histories)


# entry point, run the test harness

import tensorflow as tf

if __name__ == '__main__':
    run_test_harness()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
