# Custom datagen
import tensorflow as tf
# callbacks
# Custom datagen
from training import create_list_of_data, CustomDataGen
CHECK_POINT_FILEPATH = '/home/usafa/Documents/GitHub/automous-systems-integration-rover/model/'
DEFAULT_DATA_PATH = '/media/usafa/ext_data/data'
from sklearn.model_selection import train_test_split
INPUT_SIZE = [67,60]
df = create_list_of_data(DEFAULT_DATA_PATH, "_w")
training, test = train_test_split(df, test_size=0.2)
training_generator = CustomDataGen(DEFAULT_DATA_PATH, training, True, 5, INPUT_SIZE, 32, use_weighted=True)
test_generator = CustomDataGen(DEFAULT_DATA_PATH, test, True, 5, INPUT_SIZE, 32, use_weighted=True)

from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())


# model_path = '/home/usafa/Documents/GitHub/automous-systems-integration-rover/model/model2023_04_05_22_00_01.h5'
# model = tf.keras.models.load_model(model_path)
# model.summary()
# print(model.predict(test_generator))