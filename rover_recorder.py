import pickle

from scipy import rand

from utilities.realsense_imu import initialize_camera
from SITL_Test import connect_device

import datetime
import keyboard
import pyrealsense2.pyrealsense2 as rs
import numpy as np
import cv2
from dronekit import connect, VehicleMode, LocationGlobalRelative
import time
import logging
import random


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


def create_rand_coord():
    return [ random.randint(0, 100), random.randint(0, 100), random.randint(0, 100)]


def record(pipeline, config):
    pause = False
    i = 0
    last_frm_idx = -1

    session__id = str(datetime.datetime.now().strftime('%Y_%m_%d'))

    try:

        logging.info("Establishing Session ID:" + session__id)

        bag_name = '/media/usafa/data/data_ ' + session__id + '.bag'

        config.enable_record_to_file(f"{bag_name}")

        tele_name = '/media/usafa/data/tele_data_' + session__id + '.pkl'
        tele_data = {}

        logging.info("Configuring depth stream.")
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        logging.info("Configuring color stream.")
        config.enable_stream(rs.stream.color, rs.format.bgr8, 30)

        logging.info("Starting camera streams...")
        pipeline.start(config)

        logging.info("Recording realsense sensor stream..")

        while True:

            if not pause:
                frames = pipeline.wait_for_frames()
                bgr_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()

                cur_frm_idx = int(bgr_frame.frame_number)
                last_frm_idx = cur_frm_idx

                tele = create_rand_coord()

                tele_data[cur_frm_idx] = tele

                print(tele_data)

                if not bgr_frame or depth_frame:
                    continue

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):  # Quit program
                quit_program = True
                break

    finally:
        pipeline.stop()
        with open(tele_name, 'wb') as fp:
            pickle.dump(tele_data, fp)
            print('dictionary saved successfully to file')
            # np.save(f, tele_data)


def main():
    p = rs.pipeline()
    conf = rs.config()
    # dev = connect_device("127.0.0.1:14550")
    record(p, conf)


if __name__ == "__main__":
    # execute only if run as a script
    main()
