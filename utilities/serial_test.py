from dronekit import connect, VehicleMode, LocationGlobalRelative
import time


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


# port = "/dev/ttyTHS0"
port = "/dev/ttyUSB0"
baud = 115200  # 57600

print(f"Testing mavlink connection on port {port}, baud rate= {baud}...")

drone = connect_device(port, b=baud)

if drone is not None:
    print("Connection is successful!")
    drone.armed = False
    drone.close()
else:
    print("Connection failed.")

print("End of test.")
