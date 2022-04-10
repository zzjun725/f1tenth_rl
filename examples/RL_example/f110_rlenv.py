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


def create_f110env(no_terminal=False, env_time_limit=100000, env_action_repeat=1,
                   mapfile='./config_example_map.yaml', render=False):
    with open(mapfile) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)
    limited_time = True if env_time_limit != 0 else False
    env = F110Env_Discrete_Action(conf=conf,
                                  no_terminal=no_terminal,
                                  time_limit=env_time_limit,
                                  dictObs=True,
                                  limited_time=limited_time)
    if render:
        env.f110.add_render_callback(render_callback)
    return env


class F110Env_Discrete_Action:
    def __init__(self, speed=3, conf=None, no_terminal=None, time_limit=10000,
                 dictObs=False, limited_time=False):
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

        self.no_terminal = no_terminal
        self.action_size = self.action_space.n if hasattr(self.action_space, 'n') else env.action_space.shape[0]
        self.time_limit = time_limit
        self.dictObs = dictObs
        self.limited_time = limited_time

        self.episode = []

    def reset(self):
        starting_idx = random.sample(range(len(self.waypoints_xytheta)), 1)
        # print(self.waypoints_xytheta[starting_idx])
        starting_pos = self.waypoints_xytheta[starting_idx]
        # starting_pos[-1] += 0.5
        raw_obs, _, _, _ = self.f110.reset(starting_pos)
        # raw_obs, _, _, _ = self.f110.reset(np.array([[self.conf.sx, self.conf.sy, self.conf.stheta]]))
        obs = self.get_obs(raw_obs)

        # dict
        if self.dictObs:
            obs['action'] = np.zeros(self.action_size)
            obs['reward'] = np.array(0.0)
            obs['terminal'] = np.array(False)
            obs['reset'] = np.array(True)
            self.episode = [obs.copy()]

        # time limit
        self.step_ = 0

        return obs

    def get_action(self, action_idx: int) -> np.ndarray:
        return self.f110_action[action_idx].reshape(1, -1)

    def get_obs(self, raw_obs: dict):
        obs = raw_obs['scans'][0][::40]
        if self.dictObs:
            if len(obs.shape) == 1:
                return {'vecobs': obs}  # Vector env
            else:
                return {'image': obs}  # Image env
        else:
            return obs

    def get_reward(self, raw_obs: dict, done: bool):
        scan = raw_obs['scans'][0]
        if done:
            return -0.5
        ##  give penalty for near the wall
        if min(scan) < 0.4:
            reward = -0.02  # 0.01 -> -0.01
        else:
            reward = 0.01
        ## scale the reward
        reward *= 2

        return reward

    def step(self, action):
        # action for dict
        if self.dictObs:
            if isinstance(action, int):
                action_vec = np.zeros(self.action_size)
                action_vec[action] = 1.0
            else:
                assert isinstance(action, np.ndarray) and action.shape == (self.action_size,), "Wrong one-hot action shape"
                action_vec = action

        action = self.get_action(action)
        # print(action)
        # done = False
        raw_obs, reward, done, info = self.f110.step(action)
        info = {}
        ##  make 2 step with the same action
        # step = 3
        # while not done and step > 0:
        #     raw_obs, reward, done, info = self.f110.step(action)
        #     step -= 1
        ##  give penalty for hitting the wall
        obs = self.get_obs(raw_obs)
        reward = self.get_reward(raw_obs, done)

        # time_limit
        if self.limited_time:
            self.step_ += 1
            if self.step_ >= self.time_limit:
                done = True
                info['time_limit'] = True

        ## info
        if self.dictObs:
            obs['action'] = action_vec
            obs['reward'] = np.array(reward)
            obs['terminal'] = np.array(False if self.no_terminal else done)
            obs['reset'] = np.array(False)

            self.episode.append(obs.copy())
            if done:
                episode = {k: np.array([t[k] for t in self.episode]) for k in self.episode[0]}
                info['episode'] = episode

        return obs, reward, done, info
        # return obs, reward, done, info

    def render(self, mode='human'):
        self.f110.render(mode)

    def close(self):
        self.f110.close()


if __name__ == '__main__':
    # with open('./config_example_map.yaml') as file:
    #     conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    # conf = Namespace(**conf_dict)
    #
    # env = F110Env_Discrete_Action(conf=conf, dictObs=True)
    # env.f110.add_render_callback(render_callback)
    env = create_f110env(no_terminal=False, env_time_limit=10)

    for ep_i in range(3):
        obs = env.reset()
        done = False
        # env.render()
        i = 0
        min_obs = []
        while not done:
            i += 1
            env.render()
            steer = 0
            # speed = np.random.rand()*5
            speed = 1
            # print(speed, steer)
            # action = env.action_space.sample()
            # action = np.array([steer, speed])
            obs, step_reward, done, info = env.step(2)
            # print(obs)

            # time.sleep(1)
        #     min_obs.append(min(obs))
        #     print(step_reward)
        #     if i % 30 == 0:
        #         plt.plot(obs)
        #         plt.title(f'dimension=54')
        #         plt.show()
        print('finish one episode')
        print(info)
        # plt.plot(min_obs)
        # plt.show()
