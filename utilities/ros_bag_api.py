import os
import argparse
import  numpy as np
import cv2

import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


def main():
    """Extract a folder of images from a rosbag.
    """
    parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
    parser.add_argument("bag_file", help="Input ROS bag.")
    parser.add_argument("output_dir", help="Output directory.")
    parser.add_argument("image_topic", help="Image topic.")

    bag_file = None
    img_topic = None
    output_path = None
    args = None

    try:
        args = parser.parse_args()
        bag_file = args.bag_file
        img_topic = args.image_topic
        output_path = args.output_dir
    except SystemExit as e:
        print(f"callback exception: {e}")
        pass

    if bag_file is None:
        bag_file = '/media/internal/data/cloning_20230127-152424.bag'

    if img_topic is None:
        #img_topic =('/device_0/sensor_0/Depth_0/image/data')
        img_topic = ('/device_0/sensor_1/Color_0/image/data')

    if output_path is None:
        output_path = "/media/internal/data/cloning_20230127-152424"

    print("Extract images from %s on topic %s into %s" % (bag_file,
                                                          img_topic, output_path))

    bag = rosbag.Bag(bag_file, "r")
    bridge = CvBridge()
    count = 0

    topics = bag.get_type_and_topic_info()

    for topic, msg, t in bag.read_messages(topics=[img_topic]):
        #cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

        encoding = msg.encoding

        if encoding=="mono16":
            cv_img = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width, -1)
        else:
            cv_img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)

        file_name = output_path+f"_{msg.header.seq}"+".png"

        cv2.imwrite(file_name, cv_img)
        print("Wrote image %i" % count)

        count += 1

    bag.close()

    return


if __name__ == '__main__':
    main()
