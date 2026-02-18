# Autonomous Rover — Behavioral Cloning for Line-Following Navigation

**CNN-based imitation learning system for autonomous track navigation | USAFA Autonomous Systems Integration**

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?logo=opencv&logoColor=white)

## Overview

End-to-end behavioral cloning system that teaches a physical skid-steer rover to follow a line track using only RGB camera input. A convolutional neural network learns the mapping from camera frames to steering and throttle commands by imitating expert-driven demonstrations. Built as an autonomous systems integration project at the United States Air Force Academy.

## System Architecture

```
Intel RealSense Camera (640x480 RGB)
        │
        ▼
Image Processing (OpenCV)
  - Resize to 160x120
  - White-pixel mask extraction
  - Crop to region of interest (67x60)
  - Gaussian-weighted attention overlay
        │
        ▼
CNN Model (TensorFlow/Keras)
  - 3x Conv2D (16 filters each)
  - Dense layers: 1024 → 512 → 256 → 64 → 2
  - Output: [steering, throttle]
        │
        ▼
DroneKit MAVLink Interface
  - RC channel overrides → physical rover
```

**Data collection**: Expert drives the rover around a taped track while the system records camera frames paired with steering/throttle telemetry. Frames are masked and indexed, then stored alongside pickled telemetry arrays.

**Training**: Custom data generator feeds cropped, masked frames (with optional Gaussian-weighted channel) into the CNN. Trained with MSE loss, Adam optimizer, and early stopping on validation loss.

**Inference**: Real-time loop captures RealSense frames, processes through the trained model, and writes predicted steering/throttle values to the rover via DroneKit RC channel overrides.

## Tech Stack

- **Python 3.x**
- **TensorFlow / Keras** — CNN model definition and training
- **OpenCV** — image processing, masking, and frame capture
- **DroneKit** — MAVLink communication with the rover autopilot
- **Intel RealSense SDK** (`pyrealsense2`) — depth and color camera streams
- **NumPy / SciPy** — data processing and Gaussian kernel generation

## Project Structure

| Path | Description |
|---|---|
| `run_rover.py` | Main inference loop — camera → model → rover control |
| `training/training.py` | Model definition and training harness |
| `training/data_gen.py` | Custom data generator with sequencing support |
| `data_collection/rover_recorder.py` | Records camera frames and telemetry during expert driving |
| `data_collection/rover_data_processor.py` | Post-collection frame extraction, masking, and labeling |
| `utilities/drone_lib.py` | DroneKit helper functions |
| `utilities/realsense_imu.py` | RealSense IMU interface |
| `model/` | Saved Keras model weights (.h5) |

## Getting Started

### Hardware

- Skid-steer rover with MAVLink-compatible autopilot
- Intel RealSense depth camera
- Serial connection (`/dev/ttyACM0` at 115200 baud)

### Dependencies

```bash
pip install tensorflow opencv-python dronekit pyrealsense2 numpy scipy scikit-learn matplotlib
```

### Training

```bash
cd training
python training.py
```

### Running the Rover

```bash
python run_rover.py
```

Arm the rover via radio controller when prompted. The model takes over steering and throttle once armed.

## Context

- Autonomous Systems Integration course project — **United States Air Force Academy**, Spring 2023
- Explored 5 iterative experiments: CNN architecture tuning, data pipeline debugging, and Gaussian-weighted reinforcement signal integration

## Author

Built by Zachary Benson | Space Force Officer
[LinkedIn](https://linkedin.com/in/zacharybenson)
