import csv
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

def arm_device(device):
    while not device.is_armable:
        print("Switching device to armable...")
        time.sleep(2)
        # "GUIDED" mode sets drone to listen
        # for our commands that tell it what to do...
    while device.mode != "GUIDED":
        print("Switching to GUIDED mode...")
        device.mode = VehicleMode("GUIDED")
        time.sleep(2)
    while not device.armed:
        print("Waiting for arm...")
        time.sleep(2)
        device.armed = True


def create_rand_coord():
    return [random.randint(0, 100), random.randint(0, 100), random.randint(0, 100)]


def get_rover_data(device):
    grd_spd= device.groundspeed
    vel = device.velocity
    ster = device.channels[1]
    thr = device.channels[3]

    print(f"Ground speed:{grd_spd}")
    print(f"Velocity: {vel}")
    print(f"Steering rc: {ster}")
    print(f"Throttle rc: {thr}")

    return [grd_spd, *vel, ster, thr]


def record(pipeline, config, device):
    pause = False
    i = 0
    last_frm_idx = -1

    session__id = str(datetime.datetime.now().strftime('%Y_%m_%d_%H_%M'))

    try:

        logging.info("Establishing Session ID:" + session__id)

        bag_name = '/media/usafa/data/data_ ' + session__id + '.bag'

        config.enable_record_to_file(f"{bag_name}")

        tele_name = '/media/usafa/data/tele_data_' + session__id + '.csv'
        tele_data =  []

        logging.info("Configuring depth stream.")
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        logging.info("Configuring color stream.")
        config.enable_stream(rs.stream.color, rs.format.bgr8, 30)

        logging.info("Starting camera streams...")
        pipeline.start(config)

        logging.info("Recording realsense sensor stream..")

        while True:

            if device.armed:
                frames = pipeline.wait_for_frames()
                bgr_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()

                cur_frm_idx = int(bgr_frame.frame_number)
                last_frm_idx = cur_frm_idx

                tele = get_rover_data(device)

                # tele_data[cur_frm_idx] = tele
                tele.insert(0,cur_frm_idx)

                # tele_data.concatenate(tele)
                tele_data.append( tele)

                print(tele_data)

                if not bgr_frame or depth_frame:
                    continue
            else:
                quit_program = True
                break

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):  # Quit program
                quit_program = True
                break

    finally:
        pipeline.stop()
        # with open(tele_name, 'wb') as fp:
        #     pickle.dump(tele_data, fp)
        #     print('dictionary saved successfully to file')
        tele_data = np.array(tele_data)
        tele_data.tofile(tele_name, sep=',')


def main():
    pipeline = rs.pipeline()
    configuration = rs.config()
    rover = connect_device("127.0.0.1:14550")
    arm_device(rover)
    record(pipeline, configuration, rover)


if __name__ == "__main__":
    # execute only if run as a script
    main()
