import logging
import time

import cv2
import numpy as np
import pyrealsense2.pyrealsense2 as rs
from dronekit import connect
from tensorflow import keras

from misc.SITL_Test import connect_device

rover = None
rov_steering_val = None
rov_throttle_val = None
MODEL = '/home/usafa/Desktop/automous-systems-integration-rover-main/model/model2023_04_05_19_49_26.h5'

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

def apply_mask(color_frame):
    low_white = (150,150,150)
    high_white = (255,255,255)
# Threshold the HSV image to get only white colors
    mask = cv2.inRange(color_frame,low_white, high_white)
    # Bitwise-AND mask and original image
    return mask

def process_image(frames):


    color_frame = cv2.resize(frames, (160, 120))
    image = apply_mask(color_frame)
    height, width = image.shape
    image = image[int(width / 3):width,int(height / 2):height]

    return image

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

            image_arr = process_image(color_frame)

            logging.info("Getting predictions model..")
            image_arr = np.expand_dims(image_arr, 0)
            new_ster, new_thr = inference(image_arr, model)
            print([new_thr,new_ster])
            device.channels.overrides = {'1': int(new_ster), '3': int(new_thr)}

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
