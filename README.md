 ---

<div align="center">    
 
# Behavior Cloning :arrows_clockwise: with Rovers :red_car:
 
</div>

<div align="left">

We’re going to attempt to teach a robotic rover how to use a line as a guide for driving around a race track.  This problem will be solved using a single sensor – a simple RGB camera.  All of the control will be derived from a single image, a single frame from streamed video that represents a snapshot in time. This poses both a challenge and a great opportunity for applied machine learning.  As discussed, we will approach this problem by applying a technique called behavioral cloning – a subset under a larger group of techniques called imitation learning (IL). <br>

In general, at the heart of IL is a Markov Decision Process (MDP) where we have a set of states S, a set of actions A, and a P(s’|s,a) transition model (the probability that an action a in the state s leads to s’).  This may or may not be associated with an unknown reward function, R(s,a).  For now, there will be no R function.  If you’re thinking to yourself that this more or less resembles reinforcement learning (RL), then you’d be correct; however, our rovers will attempt to learn an expert’s optimal policy π* through a simple supervised learning approach. <br>

## Description of the Problem
The environment consists of the following: (1) State s ' S is represented by an observation through our video camera (i.e. a frame/image), along with any other data you might find useful.  Our camera has a resolution of 640x480 with 3 8-bit (RGB) color channels.  (2) A deceptively simple action space, that allows for a throttle amount and a steering amount (remember, this is a skid steer system).  The goal is to perform n laps around a track defined by colored material (some sort of matte tape) laid out in an arbitrary line pattern in the fastest time possible. Ideally, our model will generalize to drive around any shape of track. <br>


## Artifacts
### 2 - Models :link:
### 3 - Data Repo :link:
### 4 - Experiment Logs :link:

4.a Data Collection <br>
- Collected 35 minutes of data of the rover driving the track.
- Captured in this data was the video, throttle, and steering values for the rover.
    - The throttle data was later found to be incorrect, as its value is based on a relative throttle position. As such the rover throttle values, if the throttle was initialized at base line would move the rover in reverse. <br> this was addessed later on in the pipeline.
- The video data was save to bag file.
- The steering and throttle data were pickled.
  4.b Data Processing <br>
- Data processing was conducted in two phases: 1. After collection, and immediately before training.
    - After collection:
        - The video data was split into indexed individual frames.
        - A bit mask was then applied to the image to highlight the track.
        - These index frames were then matched with their telemetry <br> data that was reloaded from the pickle file as a numpy array.
        - Lastly some images during collection were corrupted, these were removed using a script <br>
          prior to being added to the data set.
- Before training:
    - The images were resized to fit the specific dimensions needed for the model. This was <br>
      accomplished by creating a custom data generator.
    - Also included in the data generator was the option to create sequences of data.

### 4.c Model Creation <br>
To create my models I conducted three primary experiments.

#### Experiment 1
- Experiment 1 Specs:
    - Input size 28 x 28
    - 3 Convulutional layers
    - 3 Drop out layers
    - 3 max pooling layers
    - Initialized with truncated normal
    - Batch normalization
    - mse as loss function

- Results: Model carried a high training loss throughout the training, but continued to be minimized.
  Validation loss quickly decreased and then remained stagnant for the remainder of the epochs.
  It was clear that this model was overfitting.
    - To verify these results, I ran an inference on one batch of the training data, and found that the throttle and steering outputs were close to the labels.
    - When this model was used on the rover it became clear that there was an issue with the data. The rover was only moving backwards. It was found that this was because the throttle values that were recorded were relative to where the throttle was set when it was turned on. Since it was not at true zero during data collection, the throttle inputs all read to be moving the rover in the reverse direction. This was addressed for the next iteration.

#### Experiment 2
- Experiment 2 Specs:
    - Input size 256 x 256
    - 3 Convolutional layers
    - 3 Drop out layers
    - 3 max pooling layers
    - Initialized with truncated normal
    - Batch normalization
    - mse as loss function

- Results: Model carried a high training loss throughout the training, but continued to be minimized.
  Validation loss quickly decreased and then remained stagnant for the remainder of the epochs.
  It was clear that this model was overfitting.
    - Due to time constraints this was tested on the rover.

#### Experiment 3
- Experiment 3 Specs:
    - Input size 256 x 256
    - 4 Convolutional layers
    - 4 Drop out layers
    - 4 max pooling layers
    - Initialized with truncated normal
    - Batch normalization
    - mse as loss function

- Results: Model carried a high training loss throughout the training, but continued to be minimized.
  Validation loss quickly decreased and then continued to steadily decrease over the epochs. Although this was true the difference between training, and val loss were quite large; this could be an indication of over fitting.
    - Due to time constraints this was tested on the rover.
 
</div>

