# SI
import pyrealsense2.pyrealsense2 as rs
import numpy as np
import cv2

"""
configure camera
"""
import datetime
import keyboard

def take_pictures():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, rs.format.bgr8, 30)
    pipeline.start(config)
    try:
        while True:
            frame = pipeline.wait_for_frames().get_color_frame()

            if frame:
                image_name = "'/media/usafa/data/test_image_" + str(datetime.datetime.now()) + ".jpg"
                img = np.asanyarray(frame.get_data())
                print('frame captured', img.shape)
                cv2.imwrite(
                    '/media/usafa/data/test_image' + str(datetime.datetime.now().strftime('%Y_%m_%d%H_%M_%S')) + ".jpg",
                    img)
            if keyboard.is_pressed("q"):
                break

    except KeyboardInterrupt:
        pass

    pipeline.stop()

def stream_video():

    try:

        config = rs.config()
        pipeline = rs.pipeline()
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
                depth_frame = frames.get_depth_frame()

                cur_frm_idx = int(color_frame.frame_number)
                if not color_frame:
                    continue

                last_frm_idx = cur_frm_idx

                color_frame = np.asanyarray(color_frame.get_data())
                depth_frame = np.asanyarray(depth_frame.get_data())

                # grab the frame dimensions and convert it to a blob
                (h, w) = color_frame.shape[:2]
                i += 1

                # show the output frame
                cv2.imshow("RGB Frame", cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR))
                cv2.imshow("Depth Frame",cv2.convertScaleAbs(depth_frame, alpha=0.03))

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):  # Quit program
                quit_program = True
                break

    finally:
        pipeline.stop()

def main():
    stream_video()


if __name__ == "__main__":
    # execute only if run as a script
    main()