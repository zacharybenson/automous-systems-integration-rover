import argparse
import pyrealsense2.pyrealsense2 as rs
import numpy as np
import cv2
import os

quit_program = False


def play_bag(file_path, no_loop=False, bgr=True):
    global quit_program

    try:

        file_name = os.path.basename(file_path)
        config = rs.config()
        rs.config.enable_device_from_file(config, file_path)
        pipeline = rs.pipeline()

        # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        if not bgr:
            config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        else:
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        pipeline.start(config)
        i = 0
        last_frm_idx = -1
        pause = False

        while True:

            if not pause:
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                cur_frm_idx = int(color_frame.frame_number)
                if not color_frame:
                    continue

                if no_loop:
                    if cur_frm_idx < last_frm_idx:
                        cv2.destroyWindow(file_name)
                        break  # video is over, exit the playback loop

                last_frm_idx = cur_frm_idx

                color_frame = np.asanyarray(color_frame.get_data())
                depth_frame = np.asanyarray(depth_frame.get_data())

                # grab the frame dimensions and convert it to a blob
                (h, w) = color_frame.shape[:2]
                i += 1

                cv2.putText(img=color_frame, text=f"frame: {cur_frm_idx}",
                            org=(10, 400), fontFace=cv2.FONT_HERSHEY_DUPLEX,
                            fontScale=1.0, color=(0, 255, 0))

                # show the output frame
                if not bgr:
                    cv2.imshow(file_name, cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR))
                    cv2.imshow("Depth Frame", cv2.convertScaleAbs(depth_frame, alpha=0.03))
                else:
                    cv2.imshow(file_name, color_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):  # Quit program
                quit_program = True
                break
            if key == ord("c") \
                    or key == ord("s"):  # close (or skip) current video
                cv2.destroyWindow(file_name)
                break
            if key == ord("p"):  # Pause current video
                if pause:
                    pause = False
                else:
                    pause = True
    finally:
        pass

    pipeline.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--input", type=str, help="Bag file to read")
    parser.add_argument("-l", "--loop", type=str, help="Set l=0 to prevent playback loop.")
    parser.add_argument("-bgr", "--bgr", type=str, help="Set bgr=1 to reverse rgb color layers during playback.")
    parser.add_argument("-ard", "--ard", type=str, help="Set ard=1 to display ArduPilot data with frames.")

    args = parser.parse_args()

    if args.loop \
            and args.loop == 0:
        no_loop = True
    else:
        no_loop = False

    if args.bgr \
            and args.bgr == '1':
        bgr = True
    else:
        bgr = False

    # TODO: enable ard feature option...

    playback_location =  "/media/usafa/data/data_ 2023_03_03.bag"

    if playback_location.upper().endswith("/ALL"):
        # Play all videos in the folder...

        # get the root folder for all videos
        playback_location = playback_location[:-4]

        # get list of files to play from root folder
        videos = os.listdir(playback_location)

        for vid_file in videos:
            if quit_program:
                break

            if vid_file.lower().endswith(".bag"):
                play_bag(os.path.join(playback_location, vid_file), no_loop=True, bgr=bgr)

        pass
    else:
        # Play specific video file
        play_bag(playback_location, no_loop=no_loop, bgr=bgr)
