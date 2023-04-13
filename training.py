# baseline cnn model for mnist
import datetime
import os

import matplotlib.pyplot as plt
import tensorflow as tf
from keras.optimizers import adam_v2
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
# callbacks
from tensorflow.keras.callbacks import ModelCheckpoint
# Custom datagen
from data_gen import create_list_of_data, CustomDataGen

INPUT_SIZE = ''
CHANNELS = ''
USE_WEIGHTED =''

#Set these before training

def initialize_training_settings(use_weighted):
    global USE_WEIGHTED
    global CHANNELS
    global INPUT_SIZE
    USE_WEIGHTED = use_weighted
    INPUT_SIZE = [67, 60]
    if use_weighted:
        CHANNELS = 2
    else:
        CHANNELS = 1


# checkpointing
CHECK_POINT_FILEPATH = '/home/usafa/Documents/GitHub/automous-systems-integration-rover/model/'
DEFAULT_DATA_PATH = '/media/usafa/ext_data/data'
session__id = str(datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))


def define_model():
    global INPUT_SIZE
    global CHANNELS
    inputs = tf.keras.Input(shape=(INPUT_SIZE[0], INPUT_SIZE[1], CHANNELS))

    x = layers.Conv2D(filters=16,
                      kernel_size=3,
                      activation="relu",
                      kernel_initializer="truncated_normal",padding='same')(inputs)
    x = layers.Conv2D(filters=16,
                      kernel_size=3,
                      activation="relu",
                      kernel_initializer="truncated_normal",padding='same')(x)
    x = layers.Conv2D(filters=16,
                      kernel_size=3,
                      activation="relu",
                      kernel_initializer="truncated_normal",padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1024, kernel_initializer="truncated_normal")(x)
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

callbacks = [model_checkpoint_callback]


# ------ Callbacks ------

# evaluate a model using k-fold cross-validation
def train(train_generator, test_generator, model, callback_list):
    histories = list()
    # fit model
    with tf.device('/GPU:0'):
        history = model.fit(train_generator,
                            epochs=100, batch_size=32,
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
    plt.plot(epochs, loss, "b", label="Training loss")
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
