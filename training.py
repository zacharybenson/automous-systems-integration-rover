# baseline cnn model for mnist
import datetime
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.optimizers import adam_v2
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
# callbacks
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ModelCheckpoint
# Custom datagen
from data_gen import create_list_of_data, CustomDataGen

global INPUT_SIZE, CHANNELS, USE_WEIGHTED

#Set these before training

def initialize_training_settings(use_weighted):
    USE_WEIGHTED = use_weighted
    if use_weighted:
        INPUT_SIZE = [160, 107]
        CHANNELS = 2
    else:
        INPUT_SIZE = [160, 107]
        CHANNELS = 1


# checkpointing
CHECK_POINT_FILEPATH = '/Users/zacharybenson/Documents/github/automous-systems-integration-rover/model/'
DEFAULT_DATA_PATH = "/Users/zacharybenson/Desktop/w/"
session__id = str(datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))


def define_model():
    inputs = tf.keras.Input(shape=(INPUT_SIZE[0], INPUT_SIZE[1], CHANNELS))

    x = layers.Conv2D(filters=16,
                      kernel_size=3,
                      activation="relu",
                      kernel_initializer="truncated_normal")(inputs)
    x = layers.Conv2D(filters=16,
                      kernel_size=3,
                      activation="relu",
                      kernel_initializer="truncated_normal")(x)
    x = layers.Conv2D(filters=16,
                      kernel_size=3,
                      activation="relu",
                      kernel_initializer="truncated_normal")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, kernel_initializer="truncated_normal")(x)
    x = layers.Dense(256, kernel_initializer="truncated_normal")(x)
    x = layers.Dense(64, kernel_initializer="truncated_normal")(x)
    outputs = layers.Dense(2)(x)


    opt = adam_v2.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, clipvalue=0.1)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss="mse", optimizer=opt)
    print(model.summary())
    return model


# ------ Callbacks ------
model_early_stoping_callback = EarlyStopping(monitor='val_loss', patience=3)
model_checkpoint_callback = ModelCheckpoint(
    filepath=CHECK_POINT_FILEPATH + "model" + session__id + ".h5",
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

callbacks = [model_early_stoping_callback,
             model_checkpoint_callback],


# ------ Callbacks ------

# evaluate a model using k-fold cross-validation
def train(train_generator, test_generator, model, callback_list):
    histories = list()
    # fit model
    with tf.device('/GPU:0'):
        history = model.fit(train_generator,
                            epochs=10, batch_size=32,
                            validation_data=test_generator,
                            callbacks=callback_list,
                            verbose=1, shuffle=True)

        model.save(CHECK_POINT_FILEPATH)
    return history
    # return score, history


# plot diagnostic learning curves
def plot_loss(histories):
    loss = histories.history["loss"]
    val_loss = histories.history["val_loss"]
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    plt.savefig('model_' + session__id + '_loss_curve' + '.png')


def load_keras_model_direct(model_path):
    model = tf.keras.models.load_model(model_path)
    model.summary()
    return model


# run the test harness for evaluating a model
def train_harness():
    # load dataset
    df = create_list_of_data(DEFAULT_DATA_PATH, "_w")
    training, test = train_test_split(df, test_size=0.2)
    training_generator = CustomDataGen(DEFAULT_DATA_PATH, training, True, 5, INPUT_SIZE, 32,use_weighted=USE_WEIGHTED)
    test_generator = CustomDataGen(DEFAULT_DATA_PATH, test, True, 5, INPUT_SIZE, 32, use_weighted=USE_WEIGHTED)

    continue_train = False

    if continue_train:
        model = load_keras_model_direct(
            '/Users/zacharybenson/Documents/github/automous-systems-integration-rover/model/model2023_04_05_16_03_49.h5')
    # evaluate model
    else:
        model = define_model()

    histories = train(training_generator, test_generator, model, callbacks)
    plot_loss(histories)


if __name__ == '__main__':
    initialize_training_settings(use_weighted=True)
    train_harness()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
