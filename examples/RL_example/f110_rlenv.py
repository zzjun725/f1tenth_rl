# -- coding: utf-8 --
from f110_gym.envs.f110_env import F110Env
from gym import spaces

import time
import yaml
import gym
import numpy as np
from argparse import Namespace
import torch

from numba import njit
from pyglet.gl import GL_POINTS

from utils import render_callback
from torch.distributions import Categorical
import matplotlib.pyplot as plt


class F110Env_Discrete_Action:
    def __init__(self, speed=5, conf=None):
        self.f110 = F110Env(map=conf.map_path, map_ext=conf.map_ext, num_agents=1)
        self.conf = conf
        self.action_space = spaces.Discrete(3)
        # steer, speed
        self.f110_action = np.array([
            [1, speed],  # go left
            [-1, speed],  # go right
            [0, speed], # go straight
            # [1, 0], # go left and reduce speed
            # [-1, 0],
            # [0, 0]
        ])
        self.observation_space = spaces.Box(low=0, high=1000, shape=(108, 1))

    def reset(self):
        conf = self.conf
        raw_obs, _, _, _ = self.f110.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
        obs = self.get_obs(raw_obs)
        return obs

    def get_action(self, action_idx:int) -> np.ndarray:
        return self.f110_action[action_idx].reshape(1, -1)

    def get_obs(self, raw_obs:dict) -> np.ndarray:
        obs = raw_obs['scans'][0][::10]
        return obs

    def step(self, action):
        action = self.get_action(action)
        raw_obs, reward, done, info = self.f110.step(action)
        if done:
            reward = -2
        obs = self.get_obs(raw_obs)
        return obs, reward, done, info

    def render(self, mode='human'):
        self.f110.render(mode)

    def close(self):
        self.f110.close()


if __name__ == '__main__':
    with open('./config_example_map.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    env = F110Env_Discrete_Action(conf=conf)
    env.f110.add_render_callback(render_callback)

    obs = env.reset()
    done = False
    env.render()
    i = 0
    while not done:
        i += 1
        env.render()
        steer = 0
        # speed = np.random.rand()*5
        speed = 5
        # print(speed, steer)
        action = 2
        obs, step_reward, done, info = env.step(action)
        if i % 30 == 0:
            plt.plot(obs)
            plt.title('dimension=1080')
            plt.show()