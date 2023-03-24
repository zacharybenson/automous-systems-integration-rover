# baseline cnn model for mnist
import keras
import tensorflow as tf

from keras.optimizers import adam_v2
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers

# callbacks
# checkpointing
CHECK_POINT_FILEPATH = '/model/model2023_03_22_11_46.h5'
DEFAULT_DATA_PATH = "/Users/zacharybenson/Desktop/data/w/"
#25 is best


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
    # opt = SGD(learning_rate=0.1, momentum=0.9)
    opt = adam_v2.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, clipvalue=0.1)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss=["mse","mse"],
                  optimizer=opt,
                  )
    print(model.summary())
    return model


model =  keras.models.load_model(CHECK_POINT_FILEPATH)
# model.load_weights(CHECK_POINT_FILEPATH)
# model.save(CHECK_POINT_FILEPATH)

