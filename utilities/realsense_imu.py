import time
import pyrealsense2.pyrealsense2 as rs
import math


def initialize_camera():
    # start the frames pipe
    p = rs.pipeline()
    conf = rs.config()
    conf.enable_stream(rs.stream.accel)
    conf.enable_stream(rs.stream.gyro)
    p.start(conf)
    print("Realsense Sensor Initialized")
    return p, conf


def cam_util_console():
    p, conf = initialize_camera()
    first = True
    alpha = 0.98

    try:
        while True:
            f = p.wait_for_frames()

            # gather IMU data
            accel = f[0].as_motion_frame().get_motion_data()
            gyro = f[1].as_motion_frame().get_motion_data()

            ts = f.get_timestamp()

            # calculation for the first frame
            if first:
                first = False
                last_ts_gyro = ts

                # accelerometer calculation
                accel_angle_z = math.degrees(math.atan2(accel.y, accel.z))
                accel_angle_x = math.degrees(math.atan2(accel.x, math.sqrt(accel.y * accel.y + accel.z * accel.z)))
                accel_angle_y = math.degrees(math.pi)

                continue

            # calculation for the second frame onwards

            # gyrometer calculations
            dt_gyro = (ts - last_ts_gyro) / 1000
            last_ts_gyro = ts

            gyro_angle_x = gyro.x * dt_gyro
            gyro_angle_y = gyro.y * dt_gyro
            gyro_angle_z = gyro.z * dt_gyro

            dangleX = gyro_angle_x * 57.2958
            dangleY = gyro_angle_y * 57.2958
            dangleZ = gyro_angle_z * 57.2958

            totalgyroangleX = accel_angle_x + dangleX
            totalgyroangleY = accel_angle_y + dangleY
            totalgyroangleZ = accel_angle_z + dangleZ

            # accelerometer calculation
            accel_angle_z = math.degrees(math.atan2(accel.y, accel.z))
            accel_angle_x = math.degrees(math.atan2(accel.x, math.sqrt(accel.y * accel.y + accel.z * accel.z)))
            accel_angle_y = math.degrees(math.pi)

            # combining gyrometer and accelerometer angles
            combinedangleX = totalgyroangleX * alpha + accel_angle_x * (1 - alpha)
            combinedangleZ = totalgyroangleZ * alpha + accel_angle_z * (1 - alpha)
            combinedangleY = totalgyroangleY

            accel_angle_z = 360 - accel_angle_z

            if accel_angle_z >= 360:
                accel_angle_z = abs(360 - accel_angle_z)

            print(
                "Angle -  X: " + str(round(combinedangleX, 2)) + "   Y: " + str(
                    round(combinedangleY, 2)) + "   Z: " + str(
                    round(combinedangleZ, 2)))
            time.sleep(2)
    except KeyboardInterrupt:
        p.stop()
        time.sleep(2)
        conf = None
        p = None

