import argparse
import pickle

import pyrealsense2.pyrealsense2 as rs
import numpy as np
import cv2
import time
import datetime
import os
import glob
from imutils.video import FPS


session__id = str(datetime.datetime.now().strftime('%Y_%m_%d'))
BAGFILENAME = 'data_ ' + session__id
NPFILENAME = 'tele_data_' + session__id
ROOT_DIR = '/media/usafa/data'
BAGFILE = ROOT_DIR + '/' + BAGFILENAME + '.bag'
NPFILE = ROOT_DIR + '/' + NPFILENAME + '.pkl'

SOURCE_PATH = "/media/8b3828b7-446b-4bae-9493-13389976bb46/rover_data/raw"
DEST_PATH = "/media/8b3828b7-446b-4bae-9493-13389976bb46/rover_data/processed"

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--input", type=str, help="Bag file to read")
args = parser.parse_args()


def load_telem_file(path):
    # Create lookup for frame index (ID)
    # In other words, come up with a way to hold frame data
    # indexed by frame ID.

    # Load data from the data file (comma delimited), and
    # hold it in a structure for quick lookup (maybe a dictionary).
    session__id = str(datetime.datetime.now().strftime('%Y_%m_%d'))
    NPFILENAME = 'tele_data_' + session__id
    ROOT_DIR = '/media/usafa/data'
    PFILE = ROOT_DIR + '/' + NPFILENAME + '.pkl'
    with open(PFILE, 'rb') as fp:
        tele = pickle.load(fp)




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


def process_bag_file(source_file, dest_folder=None, skip_if_exists=True):
    fps = None
    pipeline = None

    try:
        i = 0
        print(f"Processing {source_file}...")
        # path to file should look something like this: /media/usafa/drone_data/20210122-120614.bag
        path = args.input
        if path is None:
            path = source_file

        file_name = os.path.basename(path.replace(".bag", ""))

        if dest_folder is None:
            dest_path = os.path.join(DEST_PATH, file_name)
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
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        pipeline.start(config)
        time.sleep(1)

        # Pause here to allow pipeline to start
        # before turning off real-time streaming.
        profile = pipeline.get_active_profile()

        # Going to export depth (color and b&w),
        # rgb, and any other processed results we need/want...

        # setup colorizer for depth map
        colorizer = rs.colorizer()
        playback = profile.get_device().as_playback()
        playback.set_real_time(True)

        align_to = rs.stream.color
        alignedFs = rs.align(align_to)
        fps = FPS().start()
        while playback.current_status() == rs.playback_status.playing:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=5000)  # pipeline.poll_for_frames()
                if not frames:
                    print("no frames")
                    continue

                # align rgb to depth pixels
                aligned_frames = alignedFs.process(frames)

                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()

                # Get related throttle and steering for frame
                frm_num = color_frame.frame_number
                print(f"Processing frame {frm_num}...")
                (throttle, steering, speed) = frm_lookup.get(frm_num)
                    #frm_lookup.get(frm_num, (-1, -1, -1))
                color_frame = np.asanyarray(color_frame.get_data())
                c_depth_frame = np.asanyarray(colorizer.colorize(depth_frame).get_data())
                depth_frame = np.asanyarray(depth_frame.get_data())

                color_frame = cv2.resize(color_frame, (320, 240))

                # Maybe we want to do some white line detection here....
                white_range = None
                # white_range= based on results from processed rgb color frame
                # white_range =
                # NOTE: any cropping could be done here...
                # white_range = CROP here

                # Do we need to do any resizing, blur, edge enhancement, etc.?
                # color_frame = DO STUFF

                # What about depth... any denoising?
                # depth_frame = DO STUFF

                i += 1

                # show the output frame for sanity check
                # v2.imshow("White Out!", white_range)
                cv2.imshow("Color Processed", color_frame)
                cv2.imshow("Depth processed", depth_frame)
                # Write out our various frames
                # with data as part of the file name...

                # # rgb
                # c_frm_name = f"{'{:09d}'.format(frm_num)}_{throttle}_{steering}_c.png"

                # # white detection
                # w_frm_name = f"{'{:09d}'.format(frm_num)}_{throttle}_{steering}_w.png"

                # depth (single layer gray)
                d_frm_name = f"{'{:09d}'.format(frm_num)}_{throttle}_{steering}_d.png"

                # rgb depth
                cd_frm_name = f"{'{:09d}'.format(frm_num)}_{throttle}_{steering}_cd.png"

                # cv2.imwrite(os.path.join(dest_path, c_frm_name), color_frame)
                # cv2.imwrite(os.path.join(dest_path, w_frm_name), white_range)
                cv2.imwrite(os.path.join(dest_path, d_frm_name), depth_frame)
                cv2.imwrite(os.path.join(dest_path, cd_frm_name), c_depth_frame)
                fps.update()

                key = cv2.waitKey(1) & 0xFF

                # if the `q` key was pressed, break from the loop
                # if key == ord("q"):
                #    break
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

    print(f"Finished processing frames for {source_file}.")
    if fps is not None:
        print("Elapsed time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))


def main():
    for filename in os.listdir(ROOT_DIR):
        if filename.endswith(".bag"):
            # source_file = os.path.join(ROOT_DIR, filename)
            process_bag_file(BAGFILE)
        else:
            continue


if __name__ == "__main__":
    # for earlier datasets... not needed for new data
    # tweak_data_samples("training_data/McCurdy")

    main()
