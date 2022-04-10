"""
The system is controlled by applying a force of +1 or -1 to the cart. The pendulum starts upright, and the goal is
to prevent it from falling over.
A reward of +1 is provided for every timestep that the pole remains upright. The episode ends when the pole is more than
15 degrees from vertical, or the cart moves more than 2.4 units from the center.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import random
from torch.nn.utils import clip_grad_norm
import gym

from pydreamer import dreamer
from baselineAgents.ppo_agent import PPOAgent
from common.policys import RandomPolicy


env = gym.make('CartPole-v0')
p = RandomPolicy(env.action_space)
s = env.reset()
a, _ = p(s)
obs, reward, done, info = env.step(a)
print(info)

env.close()
