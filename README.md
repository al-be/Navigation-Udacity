# Navigation
Deep RL for navigation in Banana-Brain

# Readme

The project is about designing a Deep Reinforcement Learning (DRL) agent to navigate in an Environment to grasp yellow bananas (+1) as much as possible while avoiding blues ones (-1) in a time interval. DRL learns how to behave based on trial and error.

The state space is defined based on 37 dimensions. There are four discrete actions that set the direction of movement including moving forward, moving backward, turning left, and turning right.    

The passing score is 13 in training for a sequence of 100 episodes. The hyperparameters of DRL to pass the score asap can be set to finish around 400 episodes with average 15 while the quality of a test is 14. 



# Requirements:


The library “starr” should be installed for training.

!pip install starr

First, I tried to implement a sumtree  of “prioritized_experience_replay” based on the following links. Then, I implemented that by a different path based on the “starr” library.
 
https://github.com/Howuhh/prioritized_experience_replay/blob/main/memory/tree.py
https://github.com/rlcode/per
