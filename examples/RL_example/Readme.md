# Usage

- generate training config `get_rlConfig.py`
- train with D3QN  `train_f1tenth_dqn.py`
- evaluation with D3QN model`evaluate_f1tenth_dqn.py`



# Module Definition

- **f110_rlenv.py**

wrapped the original `f110-v0` gym environment so that this class has the standard `reset`, `step` function as the classic gym environment(make it easy to test different open source RL frame like Ray). 

classify the discrete action space. add several functions to modify the reward and observation space.

- **d3qn_agent.py**

implement an agent with double and dueling DQN.

- **train_f1tenth_dqn.py**

the entry of training or evaluation task.





# Preliminary Exploration 

## Overview

Since the reward design of current environment is quite simple(given reward for the lasting time of the car), the agent turned to learn some weird strategy like run a circle forever or stick in one position.

In that case, further adjustment about the reward and the way of sampling experience is needed for this environment. 



## Observation Space

The original observation space is a 1080-dimension np.ndarray contained all the information of the beams. However, I think those information might be redundant for our task because with lower sample rate, it is still possible to derive the relative distance of the boundary and the car. As a result, I did uniform sampling on the original observation space and reduce the dimension to 108.

<img src=".\result\ddqn\image\obs108" alt="image-20211118151713940" style="zoom:63%;" /><img src=".\result\ddqn\image\obs1080" alt="image-20211118151730035" style="zoom:63%;" />

Take an example of the beam scan information of a specific position of the car in the f1tenth gym environment. It can be shown from the picture that the features are nearly the same with uniform sampling and the curve is more smooth.



## Action Space

The original action space of f1tenth env is a continuous action space(steer, speed). After discretization, I choose three actions: turn left, turn right and keep the direction. It is worth noticing that all the actions has the same positive speed as the input since I think it is useful to begin with a more straightforward task.

I also tried adding the reduction of speed to the action space, but it turned out the agent just learned to stick in the original space without moving.



## Current Result

A typical training result is demonstrated as followed. 

<img src=".\result\ddqn\image\reward" alt="image-20211118121124500"  />

 

<img src=".\result\ddqn\image\loss" alt="image-20211118121159984"  />

Clearly this agent has not learn enough information. some model get high score in evaluation just because they keep turning a circle in the beginning.