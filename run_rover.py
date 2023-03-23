
import tensorflow.keras
from PIL.Image import Image
from tensorflow import keras
from PIL import Image as Img

from misc.SITL_Test import connect_device
import tensorflow as tf
import pyrealsense2.pyrealsense2 as rs
import numpy as np
import cv2
from dronekit import connect
import time
import logging
from tensorflow.keras.models import load_model
rover = None
rov_steering_val = None
rov_throttle_val = None
from PIL import Image as im
from training.training import define_model
MODEL = '/home/github/automous-systems-integration-rover/model2023_03_22_13_25.h5'

def connect_device(s_connection, b=115200, num_attempts=10):
    print("Connecting to device...")
    device = None
    attempts = 1

    while device is None and attempts <= num_attempts:
        print(f"  Attempt #{attempts}...")
        device = connect(s_connection, wait_ready=True, baud=b)
        time.sleep(1)
        attempts += 1

    if device is not None:
        print("Device connected.")
        print(f"Device version: {device.version}")
    else:
        print("Device not connected.")

    return device

def process_image(frames):

    color_frame= frames.get_color_frame()
    color_frame = cv2.resize(color_frame, (320, 240))

    hsv = cv2.cvtColor(color_frame, cv2.COLOR_BGR2HSV)
    sensitivity = 125
    lower_white = np.array([0, 0, 255 - sensitivity])
    upper_white = np.array([255, sensitivity, 255])

    # Threshold the HSV image to get only white colors
    mask = cv2.inRange(hsv, lower_white, upper_white)
    # Bitwise-AND mask and original image
    white_range = cv2.bitwise_and(color_frame, color_frame, mask=mask)

    # NOTE: any cropping could be done here...
    # white_range = white_range[0:214, 0:240]

    return white_range

def inference(samples, model):
    # File path


    # Convert into Numpy array

    # Generate predictions for samples
    predictions = model.predict(samples)
    predictions = predictions[0].tolist()

    return predictions

def set_rover_data(device, ster, thr):
    rov_steering_val = ster
    rov_throttle_val = thr
    print(f"Steering rc: {ster}")
    print(f"Throttle rc: {thr}")


def get_rover_data(device):
    ster = rov_steering_val
    thr = rov_throttle_val

    print(f"Steering rc: {ster}")
    print(f"Throttle rc: {thr}")

    return [ster, thr]

def percent_difference(old, new):
    return abs(old - new)/((old + new)/2) * 100


def run(pipeline, config, device,model):



    try:
        logging.info("Configuring depth stream.")
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        logging.info("Configuring color stream.")
        config.enable_stream(rs.stream.color, rs.format.bgr8, 30)

        logging.info("Starting camera streams...")
        pipeline.start(config)

        logging.info("Realsense sensor stream..")
        colorizer = rs.colorizer()
        while device.armed:

            frames = pipeline.wait_for_frames()
            align_to = rs.stream.color
            alignedFs = rs.align(align_to)
            # align rgb to depth pixels
            aligned_frames = alignedFs.process(frames)

            color_frame = aligned_frames.get_color_frame()

            color_frame = np.asanyarray(color_frame.get_data())


            color_frame = cv2.resize(color_frame, (320, 240))
            # Maybe we want to do some white line detection here....

            # define range of white color in HSV
            # change it according to your need !
            hsv = cv2.cvtColor(color_frame, cv2.COLOR_BGR2HSV)
            sensitivity = 125
            lower_white = np.array([0, 0, 255 - sensitivity])
            upper_white = np.array([255, sensitivity, 255])

            # Threshold the HSV image to get only white colors
            mask = cv2.inRange(hsv, lower_white, upper_white)
            # Bitwise-AND mask and original image
            white_range = cv2.bitwise_and(color_frame, color_frame, mask=mask)

            # NOTE: any cropping could be done here...
            # white_range = white_range[0:214, 0:240]

            # NOTE: any cropping could be done here...
            img =  Img.fromarray(white_range,'RGB')

            image = tf.image.rgb_to_grayscale(img)
            image_arr = tf.keras.preprocessing.image.img_to_array(image)
            image_arr = tf.image.resize( image_arr, (28,28)).numpy()
            image_arr = image_arr/255

            logging.info("Getting predictions model..")
            image_arr = np.expand_dims(image_arr, 0)
            new_ster, new_thr = inference(image_arr, model)
            prev_ster, prev_thr = get_rover_data(device)

            #If change is less than 20% difference then input it.
            # if percent_difference(prev_thr,new_thr) < 20.0  & percent_difference(prev_ster,new_ster) < 20.0:
            device.channels.overrides = {'1': int(new_ster), '3': int(1.1* new_thr)}

    finally:
        print("finish")



def main():
    pipeline = rs.pipeline()
    configuration = rs.config()
    rover = connect_device("/dev/ttyACM0", b=115200)
    print("Arm Device via radio controller.")
    model = keras.models.load_model(MODEL)
    while True:

        # dronkit has bugs that can pop up with newer rover firmware.
        # One of these bus is returning None for channel values!
        # This hack sets up a callback that dronekit calls into when it has new
        # channel values to relay.  We will grab the values and store globally.
        @rover.on_message('RC_CHANNELS')
        def channels_callback(self, name, message):
            global rov_steering_val, rov_throttle_val
            rov_steering_val = message.chan1_raw
            rov_throttle_val = message.chan3_raw

        while not rover.armed:
            time.sleep(1)

        print("Rover Armed... The machines are taking over...")
        run(pipeline, configuration, rover,model)


if __name__ == "__main__":
    main()
