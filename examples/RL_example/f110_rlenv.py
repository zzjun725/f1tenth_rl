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
from common.policys import GapFollowPolicy, RandomPolicy
import cv2
from time import strftime, gmtime


def create_f110env(no_terminal=False, env_time_limit=100000, env_action_repeat=1,
                   mapfile='./config_example_map.yaml', render=False, continuous_action=False):
    with open(mapfile) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)
    limited_time = True if env_time_limit != 0 else False
    if continuous_action:
        env = F110Env_Continuous_Action(conf=conf,
                                        no_terminal=no_terminal,
                                        time_limit=env_time_limit,
                                        dictObs=True,
                                        limited_time=limited_time)
    else:
        env = F110Env_Discrete_Action(conf=conf,
                                      no_terminal=no_terminal,
                                      time_limit=env_time_limit,
                                      dictObs=True,
                                      limited_time=limited_time)
    if render:
        env.f110.add_render_callback(render_callback)
    return env


class Waypoints_Manager:
    def __init__(self, wp_path=None, save_wp=False, load_wp=False) -> None:
        self.wp_path = wp_path
        self.wp = []  # (k, n)
        self.saveWp = save_wp
        self.loadWp = load_wp
        if self.saveWp:
            # self.log = open(strftime('./wp-%Y-%m-%d-%H-%M-%S',gmtime())+'.csv', 'w')
            self.log = open('./new_wp.csv', 'w')
        if self.loadWp:
            self.load_wp()


    def load_wp(self):
        with open(self.wp_path, encoding='utf-8') as f:
            self.waypoints = np.loadtxt(f, delimiter=',')
            # import ipdb; ipdb.set_trace()
            self.waypoints_xytheta = \
                np.vstack([self.waypoints[:, 0], self.waypoints[:, 1], self.waypoints[:, 2]]) 
            self.wp = self.waypoints_xytheta  # (3, n)
    
    def draw_wp(self):
        plt.plot(self.wp[0], self.wp[1], '-ro', markersize=0.1)
        plt.show()

    def get_nearest_wp(self, cur_position):
        # position: (x, y)
        wp_xyaxis = self.wp[:2]  # (2, n)
        dist = np.linalg.norm(wp_xyaxis-cur_position.reshape(2, 1), axis=0)
        nearst_idx = np.argmin(dist)
        nearst_point = wp_xyaxis[:, nearst_idx]
        return nearst_point

    def get_lateral_error(self, raw_obs, ego_idx=0):
        # pose: (x, y, yaw)
        x, y, theta = raw_obs['poses_x'][ego_idx], raw_obs['poses_y'][ego_idx], raw_obs['poses_theta'][ego_idx]
        pose = np.array([x, y, theta])
        yaw = pose[2]
        local2global = np.array([[np.cos(yaw), -np.sin(yaw), 0, pose[0]], 
                                 [np.sin(yaw), np.cos(yaw), 0, pose[1]], 
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]])        
        
        # wp_xyaxis = self.wp[:2]
        nearstP = self.get_nearest_wp(pose[:2])

        global2local = np.linalg.inv(local2global)
        nearstP_local = global2local @ np.array([nearstP[0], nearstP[1], 0, 1]) 
        cur_error = nearstP_local[1]
        return abs(cur_error)
    
    def save_wp(self, raw_obs, ego_idx=0):
        x, y, theta = raw_obs['poses_x'][ego_idx], raw_obs['poses_y'][ego_idx], raw_obs['poses_theta'][ego_idx]
        self.log.write('%f, %f, %f\n' % (x, y, theta))



class F110Env_RL:
    def __init__(self, speed=3, obs_shape = 54, conf=None, no_terminal=None, time_limit=10000,
                 dictObs=False, limited_time=False) -> None:
        self.f110 = F110Env(map=conf.map_path, map_ext=conf.map_ext, num_agents=1)
        self.conf = conf
        self.speed = speed
        self.observation_space = spaces.Box(low=0, high=1000, shape=(obs_shape, 1))
        
        # waypoints Manager, for lateral error
        self.wpManager = Waypoints_Manager(conf.wpt_path, save_wp=False, load_wp=True)
        self.wpManager.load_wp()
        self.waypoints_xytheta = self.wpManager.wp.T
        self.lateral_error_thres = 0.2

        # for offline data storage
        self.no_terminal = no_terminal
        self.action_size = self.action_space.n if hasattr(self.action_space, 'n') else self.action_space.shape[0]
        self.time_limit = time_limit
        self._max_episode_steps = time_limit
        self.dictObs = dictObs
        self.limited_time = limited_time

        self.episode = []

    def reset(self):
        starting_idx = random.sample(range(len(self.waypoints_xytheta)), 1)
        # print(self.waypoints_xytheta[starting_idx])
        x, y = self.waypoints_xytheta[starting_idx][0, 0], self.waypoints_xytheta[starting_idx][0, 1]  # because self.waypoints_xytheta[starting_idx] has shape(1,3)
        # theta = 2*random.random() - 1
        theta = self.waypoints_xytheta[starting_idx][0, 2]
        starting_pos = np.array([[x, y, theta]])
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
            obs['raw_obs'] = raw_obs
            self.episode = [obs.copy()]

        # time limit
        self.step_ = 0

        return obs
    
    def get_obs(self, raw_obs: dict):
        obs = raw_obs['scans'][0][::20]
        if self.dictObs:
            if len(obs.shape) == 1:
                return {'vecobs': obs}  # Vector env
            else:
                return {'image': obs}  # Image env
        else:
            return obs
    
    def get_reward(self, raw_obs: dict, crash: bool):
        if 'raw_obs' in raw_obs.keys():
            raw_obs = raw_obs['raw_obs']
        lateral_error = self.wpManager.get_lateral_error(raw_obs)
        if crash:
            reward = -0.05
        if lateral_error > self.lateral_error_thres:
            reward = 0
            reward - lateral_error*0.01
        else:
            reward = 0.02
        return reward

    
    def render(self, mode='human'):
        self.f110.render(mode)

    def close(self):
        self.f110.close()
    
    def get_action(self):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError


class F110Env_Discrete_Action(F110Env_RL):
    def __init__(self, speed=3, obs_shape = 27, conf=None, no_terminal=None, time_limit=10000,
                 dictObs=False, limited_time=False):
        self.action_space = spaces.Discrete(3)
        super().__init__(speed, obs_shape, conf, no_terminal, time_limit, dictObs, limited_time)
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


    def get_action(self, action_idx: int) -> np.ndarray:
        return self.f110_action[action_idx].reshape(1, -1)

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
        raw_obs, reward, done, info = self.f110.step(action)
        info = {}
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
            obs['raw_obs'] = raw_obs

            self.episode.append(obs.copy())
            if done:
                episode = {k: np.array([t[k] for t in self.episode]) for k in self.episode[0]}
                info['episode'] = episode

        return obs, reward, done, info
        # return obs, reward, done, info


########### Bowen Jiang, Added on Apr 17, 2022
class F110Env_Continuous_Action(F110Env_RL):
    def __init__(self, speed=3, obs_shape = 54, conf=None, no_terminal=None, time_limit=100000,
                 dictObs=False, limited_time=False):
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,))      # in radians
        super().__init__(speed, obs_shape, conf, no_terminal, time_limit, dictObs, limited_time)
        ## TODO: fix this
        self.action_size = 2
        
    def get_action(self, action) -> np.ndarray:
        # if type(action) != int:
        #     return action.reshape(1, -1)
        # steer = np.clip(action, a_min=-1, a_max=1)[0]
        steer = np.clip(action, a_min=-1, a_max=1)
        action = np.array([steer, self.speed])
        return action.reshape(1, -1)

    def step(self, action):
        action = self.get_action(action)
        # import ipdb
        # ipdb.set_trace()
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
            obs['action'] = action
            obs['reward'] = np.array(reward)
            obs['terminal'] = np.array(False if self.no_terminal else done)
            obs['reset'] = np.array(False)
            obs['raw_obs'] = raw_obs
            # import ipdb
            # ipdb.set_trace()

            self.episode.append(obs.copy())
            if done:
                # import ipdb
                # ipdb.set_trace()
                episode = {k: np.array([t[k] for t in self.episode]) for k in self.episode[0]}
                info['episode'] = episode

        return obs, reward, done, info
        # return obs, reward, done, info

###########


def test_env(debug=False):
    env = create_f110env(no_terminal=False, env_time_limit=0, render=True, continuous_action=True)
    policy = GapFollowPolicy()
    wp_manager = Waypoints_Manager(save_wp=False)
    W, H = 1200, 500
    scan_drawScale = 40

    for ep_i in range(5):
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

            action, metric = policy(obs['raw_obs'])
            
            # print(best_p_idx)

            # print(scan.shape)
            
            # draw LiDAR
            if debug:
                best_p_idx = metric['target_idx']
                scan = (obs['raw_obs']['scans'][0]*scan_drawScale).astype(np.int64)
                scan = np.vstack([np.arange(len(scan)).astype(np.int64), 500-scan]).T
                target = scan[max(0, best_p_idx-10):best_p_idx + 10, :]
                blank_image = np.ones((H, W, 3), np.uint8)
                blank_image.fill(0)
                cv2.polylines(blank_image, [scan], False, (0, 0, 255))           
                cv2.polylines(blank_image, [target], False, (0, 255, 0))
                cv2.imshow('debug', blank_image)
                cv2.waitKey(5)

            # print(action)
            obs, step_reward, done, info = env.step(action[0])
            if i % 10 == 0:
                # print(f'step_reward: {step_reward}')
                # print(min(obs['vecobs']))
                # print(env.wpManager.get_lateral_error(obs['raw_obs']))
                print(step_reward)
                pass
                
            # if i > 100 and i % 5 == 0:
            #     wp_manager.save_wp(obs['raw_obs'])
            # print(done)
            # print(obs)

            # time.sleep(1)
        #     min_obs.append(min(obs))
        #     print(step_reward)
        #     if i % 30 == 0:
        #         plt.plot(obs)
        #         plt.title(f'dimension=54')
        #         plt.show()
        print('finish one episode')



if __name__ == '__main__':
    # wp_manager = Waypoints_Manager(wp_path='./new_wp.csv', save_wp=False, load_wp=True)
    # wp_manager.draw_wp()
    test_env(debug=True)