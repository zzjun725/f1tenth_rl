# Usage

- generate training config `get_rlConfig.py`
- train/ evaluation with D3QN  `train_f1tenth_dqn.py`
- train/ evaluation with PPO `train_f1tenth_ppo.py`

# Module Definition

- **f110_rlenv.py**

wrapped the original `f110-v0` gym environment so that this class has the standard `reset`, `step` function as the classic gym environment(make it easy to test different open source RL frame like Ray). 

classify the discrete action space and add several functions to modify the reward and observation space.

- **d3qn_agent.py**

implement an agent with double and dueling DQN.

- **ppo_agent.py**

implement a ppo agent with GAE as the metric of advantage funciton. 

- **train_f1tenth_dqn.py**

the entry of training or evaluation task.



# Training with PPO

## Overview of results

After some fine-tuning on the hyper-parameters and adjustments on the environment design,  The trained agent is able to run over the given map without collision. 

<video src=".\result\others\un_collision_model1.mp4"></video>

In multiple experiments with similar parameters, after training for a lot episodes the car seems to converge to a 'wiggle' between left and right even when it is racing on a straight road.

Besides, an agent which generate smooth path might occur in the middle of training and it already had a fair performance. The video below illustrated an agent which only fail to pass one corner in the whole map. 

<video src=".\result\others\smooth_path.mp4"></video>



## Adjustment on Environment

### Observation Space

The original observation space is a 1080-dimension np.ndarray contained all the information of the beams. However, I think those information might be redundant for our task because with lower sample rate, it is still possible to derive the relative distance of the boundary and the car. As a result, I did uniform sampling on the original observation space and reduce the dimension to 27.

<img src=".\result\ddqn\image\obs108" alt="image-20211118151713940" style="zoom:63%;" /><img src=".\result\ddqn\image\obs1080" alt="image-20211118151730035" style="zoom:63%;" />

<img src=".\result\others\obs27.PNG" alt="obs27" style="zoom:61%;" /><img src=".\result\others\obs54.PNG" alt="obs54" style="zoom: 63%;" />

Take an example of the beam scan information of a specific position of the car in the f1tenth gym environment. It can be shown from the picture that the features are nearly the same with uniform sampling and the curve is more smooth.



### Action Space

The original action space of f1tenth env is a continuous action space(steer, speed). After discretization, I choose three actions: turn left, turn right and keep the direction. It is worth noticing that all the actions has the same positive speed as the input. since the current goal is keeping the car on the right track without collision, I think it is useful to begin with a more straightforward task.

I also tried adding the reduction of speed to the action space, but it turned out the agent just learned to stick in the original space without moving. Besides, if  two speed transmissions are set, the agent will converge to choose the action which has lower speed. I think it is natural because high speed is more easily to have collision.

```python
# the final action space of the car
# speed=3
self.f110_action = np.array([
    [1, speed],  # go left
    [-1, speed], # go right
    [0, speed],  # go straight
])
```



### Modify the Step and reset function

To make the collection of training data more efficient, I stack three steps in the simulation environment as a single step in the RL-env with the same action. Besides, instead of reset the car in the same position every time, I sampled a random point on the track as the initial position so that it would not stuck in useless experiences in the beginning.



### Reward

I changed a little terms on the reward design, including penalty on hitting the wall as well as being too close to the wall, and making a scaling on the final step reward.



## Training

 The hyper-parameters of the PPO agent are as follows:

| Lr    | eps_clip | gamma | gae_lambda | epochs_per_update | step_per_update | episode |
| ----- | -------- | ----- | ---------- | ----------------- | --------------- | ------- |
| 0.001 | 0.2      | 0.99  | 0.95       | 3                 | 100             | 5000    |

I set the maximum episode to 5000, but usually 600-700 episodes are enough for training a fair agent since the goal is quite straight.

After several training test, I find out the success rate is highly depending on the exploration in the beginning. For example, it might be the case that the agent is stuck in a local minima of turning a circle or just keeping going straight whatever the environments are. Adding the penalty on hitting the wall as well as being too close to the wall alleviate this situation.

![ppo_cyclefail](D:\Code_Projects\f1tenth_gym\examples\RL_example\result\others\ppo_cyclefail.PNG)

This curve above demonstrated a typical result of training an agent which just turn around all the time until it hit the wall.

 As for a success trained model, the curve of score is more bouncing with a obvious rising trend:

![ppo_speed4_action3_success](.\result\others\ppo_speed4_action3_success.PNG)

It is worth noting that during the training process, I have observed several interesting behaviors of the agent.

- Turning around

<video src=".\result\others\tuningaround.mp4"></video>

- Turning back

<video src=".\result\others\turning_back.mp4"></video>

However, the best model always tend to has a 'wiggle' feature during the whole time, maybe taking the action of turning left or right gives the car more opportunities to pass the corner.

<video src=".\result\others\wiggle_2.mp4"></video>





# Training with DQN

When it comes to DQN, the result is not as good as PPO. After converge, the agent just tend to turn around forever or do nothing but go straight. I think more adjustments may be needed training with DQN to get a fair result.

A typical training result is demonstrated as followed. 

<img src=".\result\ddqn\image\reward" alt="image-20211118121124500"  />

 

<img src=".\result\ddqn\image\loss" alt="image-20211118121159984"  />

