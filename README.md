# Environment
The environment used by this project is forked from the [f1tenth environment](https://github.com/f1tenth/f1tenth_gym).

Check cnn_dreamer branch for latest code(kind of rough though:(

Install the environment: inside this repo, run 

`pip3 install --user -e gym/`

Install other requirements:

`pip install -r requirements.txt` 

Note: logging dependency(such as mlflow) is for dreamer training. If you just want to evaluate the model provided in evaluate_model, you can skip that

# Usage
All the codes related is under example/RL_example. To run on your computer, you need to re-generate the config files to solve the path dependency.
- visualize environment with LiDAR and distance reconstruction([video](https://drive.google.com/file/d/1pf5z0KcnPATvTQf8BxCaICx7zfznoXmu/view?usp=sharing)) `example/RL_example/f110_rlenv.py`
- generate training config `example/RL_example/config/get_rlConfig.py`
- train with D3QN  `example/RL_example/train_f1tenth_dqn.py`
- train with PPO `example/RL_example/train_f1tenth_ppo.py`
- evaluate with PPO `example/RL_example/evaluate_f1tenth_ppo.py`
Note that evaluate with PPO provide both continuous([video](https://drive.google.com/file/d/1AcNPpWkZnkifcYpcsxazveSrb5Wwgmq7/view?usp=sharing)) and discrete environment([video](https://drive.google.com/file/d/1OsrkJUFsln1yM19JoXoBbK6iBL2oxFiM/view?usp=sharing)).
- evaluate and reconstruction from latent space with dreamer `example/RL_example/evaluate_f1tenth_dream.py`
Note that the model of dreamer is too big, so we stored it in [google drive](https://drive.google.com/file/d/19o6du-l7uLfSlpW7HsG79sze7Sg-QSRY/view?usp=sharing)

# Module Definition

- **f110_rlenv.py**

Wrapped the original `f110-v0` gym environment so that this class has the standard `reset`, `step` function as the classic gym environment. 

Classify the discrete and continuous action space and add several functions to modify the reward and observation space.

Add scan and waypoint manager classes for better visualization and training performance.

- **baselineAgents**

Including the implementation of all the baseline agents such as ppo(for both continuous and discrete space) and dqn.

- **evaluate_model**

Including all the models for evaluation.

- **pydreamer**

This part of code is refered to [pydreamer](https://github.com/jurgisp/pydreamer)

# Contributions

Bowen Jiang: PPO with continuous action space, LiDAR distance reconstruction, environment design

Zhiyang Chen: DQN with discrete action space, reward design and expert training data collection

Zhijun Zhuang: PPO with discrete action space, Dreamer(only training and evaluation part), environment design
