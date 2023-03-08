 
---

<div align="center">    
 
# Behavior Cloning :arrows_clockwise: with Rovers :red_car:

## Purpose & Objectives
We’re going to attempt to teach a robotic rover how to use a line as a guide for driving around a race track.  This problem will be solved using a single sensor – a simple RGB camera.  All of the control will be derived from a single image, a single frame from streamed video that represents a snapshot in time. This poses both a challenge and a great opportunity for applied machine learning.  As discussed, we will approach this problem by applying a technique called behavioral cloning – a subset under a larger group of techniques called imitation learning (IL). 

In general, at the heart of IL is a Markov Decision Process (MDP) where we have a set of states S, a set of actions A, and a P(s’|s,a) transition model (the probability that an action a in the state s leads to s’).  This may or may not be associated with an unknown reward function, R(s,a).  For now, there will be no R function.  If you’re thinking to yourself that this more or less resembles reinforcement learning (RL), then you’d be correct; however, our rovers will attempt to learn an expert’s optimal policy π* through a simple supervised learning approach.

## Description of the Problem
The environment consists of the following: (1) State s ' S is represented by an observation through our video camera (i.e. a frame/image), along with any other data you might find useful.  Our camera has a resolution of 640x480 with 3 8-bit (RGB) color channels.  (2) A deceptively simple action space, that allows for a throttle amount and a steering amount (remember, this is a skid steer system).  The goal is to perform n laps around a track defined by colored material (some sort of matte tape) laid out in an arbitrary line pattern in the fastest time possible. Ideally, our model will generalize to drive around any shape of track.

## Artifacts
### 2 - Models :link:
### 3 - Data Repo :link:
### 4 - Experiment Logs :link:
4.a Data Processing <br>
4.b Data Collection <br>
4.c Model Creation <br>
    
### 5 - Contributers and Acknowledements



