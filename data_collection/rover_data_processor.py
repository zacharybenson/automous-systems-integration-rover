import pickle
import cv2
import cv2
import time
import datetime
import os
import glob
from imutils.video import FPS
import pyrealsense2.pyrealsense2 as rs
import numpy as np

# 10.1.100.236 accer

session__id = str(datetime.datetime.now().strftime('%Y_%m_%d_%H_%M'))
ROOT_DIR = '/media/usafa/data/'


def load_telem_file(path):
    # Load data from the data file (comma delimited), and
    # hold it in a structure for quick lookup (maybe a dictionary).
    with open(path, "rb") as fp:
        tele = pickle.load(fp)

    return tele


def get_num_frames(filename):
    # returns number of frames in bag data stream.
    cfg = rs.config()
    cfg.enable_device_from_file(filename)

    # setup pipeline for the bag file
    pipe = rs.pipeline()

    # start streaming from file
    profile = pipe.start(cfg)

    # setup playback
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)

    # get the duration of the video
    t = playback.get_duration()

    # compute the number of frames (30fps setting)
    frame_counts = t.seconds * 30
    playback.pause()
    pipe.stop()
    return t, frame_counts


def apply_mask(color_frame):
    hsv = cv2.cvtColor(color_frame, cv2.COLOR_BGR2HSV)
    sensitivity = 85
    lower_white = np.array([0, 0, 255 - sensitivity])
    upper_white = np.array([255, sensitivity, 255])

    # Threshold the HSV image to get only white colors
    mask = cv2.inRange(hsv, lower_white, upper_white)
    # Bitwise-AND mask and original image
    return mask


def crop_image(image):
    width, height = image.size

    # Setting the points for cropped image
    left = width / 3
    right = width

    top = height / 2
    bottom = height

    # Cropped image of above dimension
    # (It will not change original image)
    image_cropped = image.crop((left, top, right, bottom))

def process_bag_file(path, dest_folder=None, skip_if_exists=False):
    fps = None
    pipeline = None

    try:
        i = 0
        file_name = os.path.basename(path.replace(".bag", ""))
        print(f"Processing {file_name}...")

        if dest_folder is None:
            dest_path = os.path.join(ROOT_DIR, file_name)
        else:
            dest_path = os.path.join(dest_folder, file_name)

        # Are we skipping previously processed files?
        if skip_if_exists:
            if os.path.isdir(dest_path):
                print(f"{file_name} was previously processed; skipping file...")
                return

        # Make subfolder to hold all training data
        os.makedirs(dest_path, exist_ok=True)

        # Load data associated with the video.
        frm_lookup = load_telem_file(path.replace(".bag", ".pkl"))

        # duration, frame_count = get_num_frames(path)

        config = rs.config()

        rs.config.enable_device_from_file(config, path, False)
        pipeline = rs.pipeline()

        # Enable both color and depth streams
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, rs.format.bgr8, 30)

        pipeline.start(config)
        time.sleep(1)

        # Pause here to allow pipeline to start
        # before turning off real-time streaming.
        profile = pipeline.get_active_profile()

        # setup colorizer for depth map
        colorizer = rs.colorizer()
        playback = profile.get_device().as_playback()
        playback.set_real_time(True)

        align_to = rs.stream.color
        alignedFs = rs.align(align_to)
        fps = FPS().start()
        while playback.current_status() == rs.playback_status.playing:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=5000)
                if not frames:
                    print("no frames")
                    continue

                # align rgb to depth pixels
                aligned_frames = alignedFs.process(frames)

                color_frame = aligned_frames.get_color_frame()

                # Get related throttle and steering for frame
                frm_num = color_frame.frame_number
                print(f"Processing frame {frm_num}...")
                (throttle, steering, _, _, _, _) = frm_lookup.get(frm_num)

                color_frame = np.asanyarray(color_frame.get_data())

                color_frame = cv2.resize(color_frame, 160, 120)

                white_frame = apply_mask(color_frame)

                white_frame = crop_image(white_frame)

                i += 1

                w_frm_name = f"{'{:09d}'.format(frm_num)}_{throttle}_{steering}_w.png"

                cv2.imwrite(os.path.join(dest_path, w_frm_name), white_frame)

                fps.update()

                key = cv2.waitKey(1) & 0xFF

            except Exception as e:
                print(e)
                continue
    except Exception as e:
        print(e)
    finally:
        pass
    try:

        # stop recording
        if fps is not None:
            fps.stop()
        time.sleep(0.5)
        if playback is not None:
            if playback.current_status() == rs.playback_status.playing:
                playback.pause()
                if pipeline is not None:
                    pipeline.stop()
                    time.sleep(0.5)
    except Exception as e:
        print("Unexpected error during cleanup.", exc_info=True)

    playback = None
    pipeline = None

    print(f"Finished processing frames for {file_name}.")
    if fps is not None:
        print("Elapsed time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))


def main():
    for filename in os.listdir(ROOT_DIR):
        if filename.endswith(".bag"):
            process_bag_file(BAGFILE)
        else:
            continue


if __name__ == "__main__":
    main()
