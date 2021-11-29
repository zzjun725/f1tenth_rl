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
import random
from math import pi


class F110Env_Discrete_Action:
    def __init__(self, speed=3, conf=None):
        self.f110 = F110Env(map=conf.map_path, map_ext=conf.map_ext, num_agents=1)
        self.conf = conf
        self.speed = speed
        self.action_space = spaces.Discrete(3)
        # steer, speed
        self.f110_action = np.array([
              # go straight
            [1, speed],  # go left
            [-1, speed], # go right
            [0, speed],
            # [1, speed/2], # go left and reduce speed
            # [-1, speed/2],
            # [0, speed/2]
        ])
        self.observation_space = spaces.Box(low=0, high=1000, shape=(27, 1))

        # load waypoints
        with open(conf.wpt_path, encoding='utf-8') as f:
            self.waypoints = np.loadtxt(f, delimiter=';')
            self.waypoints_xytheta = \
                np.vstack([self.waypoints[:, 1], self.waypoints[:, 2], self.waypoints[:, 3] + pi / 2]).T

    def reset(self):
        starting_idx = random.sample(range(len(self.waypoints_xytheta)), 1)
        # print(self.waypoints_xytheta[starting_idx])
        starting_pos = self.waypoints_xytheta[starting_idx]
        # starting_pos[-1] += 0.5
        raw_obs, _, _, _ = self.f110.reset(starting_pos)
        # raw_obs, _, _, _ = self.f110.reset(np.array([[self.conf.sx, self.conf.sy, self.conf.stheta]]))
        obs = self.get_obs(raw_obs)
        return obs

    def get_action(self, action_idx: int) -> np.ndarray:
        return self.f110_action[action_idx].reshape(1, -1)

    def get_obs(self, raw_obs: dict) -> np.ndarray:
        obs = raw_obs['scans'][0][::40]
        return obs

    def step(self, action):
        action = self.get_action(action)
        # print(action)
        ##  make 2 step with the same action
        done = False
        step = 3
        while not done and step > 0:
            raw_obs, reward, done, info = self.f110.step(action)
            step -= 1
        ##  give penalty for hitting the wall
        if done:
            reward = -0.5

        ##  give penalty for near the wall
        obs = self.get_obs(raw_obs)
        if min(obs) < 0.4:
            reward -= 0.02  # 0.01 -> -0.01

        ## scale reward
        reward *= 2
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

    for ep_i in range(10):
        obs = env.reset()
        done = False
        env.render()
        i = 0
        min_obs = []
        while not done:
            i += 1
            env.render()
            steer = 0
            # speed = np.random.rand()*5
            speed = 3
            # print(speed, steer)
            # action = env.action_space.sample()
            # action = np.array([steer, speed])
            obs, step_reward, done, info = env.step(2)
            min_obs.append(min(obs))
            print(step_reward)
            if i % 30 == 0:
                plt.plot(obs)
                plt.title(f'dimension=54')
                plt.show()
        print('finish one episode')
        # plt.plot(min_obs)
        # plt.show()
