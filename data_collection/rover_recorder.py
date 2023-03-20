import pickle

from misc.SITL_Test import connect_device

import datetime
import pyrealsense2.pyrealsense2 as rs
from dronekit import connect
import time
import logging

rover = None
rov_steering_val = None
rov_throttle_val = None


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


def get_rover_data(device):
    ster = rov_steering_val
    thr = rov_throttle_val
    grd_spd = device.groundspeed
    vel = device.velocity

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
    print("Recording for session id " + session__id)

    tele_name = '/media/usafa/data/tele_data_' + session__id + '.pkl'
    tele_data = {}

    try:

        logging.info("Establishing Session ID:" + session__id)

        bag_name = '/media/usafa/data/data_' + session__id + '.bag'

        config.enable_record_to_file(f"{bag_name}")

        logging.info("Configuring depth stream.")
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        logging.info("Configuring color stream.")
        config.enable_stream(rs.stream.color, rs.format.bgr8, 30)

        logging.info("Starting camera streams...")
        pipeline.start(config)

        logging.info("Recording realsense sensor stream..")

        while device.armed:
            frames = pipeline.wait_for_frames()
            bgr_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            cur_frm_idx = int(bgr_frame.frame_number)
            last_frm_idx = cur_frm_idx

            tele = get_rover_data(device)

            tele_data[cur_frm_idx] = tele

    finally:
        pipeline.stop()
        with open(tele_name, 'wb') as fp:
            pickle.dump(tele_data, fp)
            print('dictionary saved successfully to file')


def main():
    while True:
        pipeline = rs.pipeline()
        configuration = rs.config()
        rover = connect_device("/dev/ttyACM0", b=115200)
        print("Arm Device via radio controller.")

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

        print("Rover Armed... Recording Starting.")
        record(pipeline, configuration, rover)


if __name__ == "__main__":
    # execute only if run as a script
    main()
