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
                   mapfile='./config_example_map.yaml', render=False, continuous_action=False, display_lidar=False):
    with open(mapfile) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)
    limited_time = True if env_time_limit != 0 else False
    if continuous_action:
        env = F110Env_Continuous_Action(conf=conf,
                                        no_terminal=no_terminal,
                                        time_limit=env_time_limit,
                                        dictObs=True,
                                        limited_time=limited_time,
                                        display_lidar=display_lidar)
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

class Lidar_Manager:
    def __init__(self, window_W=1080+250, window_H=250, scanScale=10):
        """
        Include the code of displays the Lidar scan and reconstruction using cv2
        """
        self.map_size = window_H
        self.map = np.zeros((self.map_size, self.map_size))

        self.lidar_dmin = 0
        self.lidar_dmax = 30
        self.angle_min = -135
        self.angle_max = 135
        self.x_min, self.x_max = -30, 30
        self.y_min, self.y_max = -30, 30
        self.resolution = 0.25
        self.interpolate_or_not = False

        # windows
        self.scanScale = scanScale
        self.window_H = window_H
        self.window_W = window_W
        self.lidar_scanPic = np.zeros((self.window_H, 1080, 3), np.uint8)
        self.lidar_reconstructPic = np.zeros((self.map_size, self.map_size, 3), np.uint8)
        # import ipdb;ipdb.set_trace()
        self.lidar_window = np.hstack([self.lidar_scanPic, self.lidar_reconstructPic])
        print(f'lidar_window shape{self.lidar_window.shape}')

    def rays2world(self, distance):
		# convert lidar scan distance to 2d locations in space
        angles = np.linspace(self.angle_min, self.angle_max, self.dimension) * np.pi / 180
        x = distance * np.cos(angles)
        y = distance * np.sin(angles)
        return x, y

    def grid_cell_from_xy(self, x, y):
		# convert 2d locations in space to 2d array coordinates
        x = np.clip(x, self.x_min, self.x_max)
        y = np.clip(y, self.y_min, self.y_max)

        cell_indices = np.zeros((2, x.shape[0]), dtype='int')
        cell_indices[0, :] = np.floor((x - self.x_min) / self.resolution)
        cell_indices[1, :] = np.floor((y - self.y_min) / self.resolution)
        return cell_indices

    def interpolate(self, cell_indices):
        for i in range(cell_indices.shape[1] - 1):
            fill_x = np.linspace(cell_indices[1, i], cell_indices[1, i+1], endpoint=False, dtype='int')
            fill_y = np.linspace(cell_indices[0, i], cell_indices[0, i+1], endpoint=False, dtype='int')
            self.map[fill_x, fill_y] = 1

    def update_scan2map(self, lidar_1d):
        self.map = np.zeros((self.map_size, self.map_size))
        self.lidar_reconstructPic = np.zeros((self.map_size, self.map_size, 3), np.uint8)

        self.distance = lidar_1d
        self.dimension = len(self.distance)

        x, y = self.rays2world(self.distance)
        cell_indices = self.grid_cell_from_xy(x, y)
        self.map[cell_indices[1, :], cell_indices[0, :]] = 1

        if self.interpolate_or_not:
        	self.interpolate(cell_indices[:, :])
        
        cell_indices_line = np.vstack([cell_indices[0,:], cell_indices[1, :]]).T
        cv2.polylines(self.lidar_reconstructPic, [cell_indices_line], False, (0, 0, 255), 5)
        # self.lidar_reconstructPic[:, :, 2][np.nonzero(self.map)[0], np.nonzero(self.map)[1]] = 200
        # import ipdb; ipdb.set_trace()
		# plt.imshow(self.map)
		# plt.show()

    def update_scan(self, best_p_idx=None, scan=None):
        self.lidar_scanPic = np.zeros((self.window_H, 1080, 3), np.uint8)
        scan = (scan*self.scanScale).astype(np.int64)
        scan = np.vstack([np.arange(len(scan)).astype(np.int64), self.window_H-scan]).T

        cv2.polylines(self.lidar_scanPic, [scan], False, (0, 0, 255), 10)
        if best_p_idx:  
            target = scan[max(0, best_p_idx-1):best_p_idx + 1, :]     
            cv2.polylines(self.lidar_scanPic, [target], False, (0, 255, 0), 10)

    
    def update_lidar_windows(self, wait=1, obs=None, target_idx=None):
        if 'raw_obs' in obs.keys():
            obs = obs['raw_obs']
        scan = obs['scans'][0]
        self.update_scan2map(scan)
        self.update_scan(best_p_idx=target_idx, scan=scan)
        self.lidar_window = np.hstack([self.lidar_scanPic, self.lidar_reconstructPic])
        cv2.imshow('debug', self.lidar_window)
        cv2.waitKey(wait)


class F110Env_RL:
    def __init__(self, speed=3, obs_shape = 54, conf=None, no_terminal=None, time_limit=10000,
                 dictObs=False, limited_time=False, display_lidar=False) -> None:
        self.f110 = F110Env(map=conf.map_path, map_ext=conf.map_ext, num_agents=1)
        self.conf = conf
        self.speed = speed
        self.observation_space = spaces.Box(low=0, high=1000, shape=(obs_shape, 1))
        
        # waypoints Manager, for lateral error
        self.wpManager = Waypoints_Manager(conf.wpt_path, save_wp=False, load_wp=True)
        self.wpManager.load_wp()
        self.waypoints_xytheta = self.wpManager.wp.T
        self.lateral_error_thres = 0.2

        # lidar
        self.display_lidar = display_lidar 
        if display_lidar:
            self.lidarManager = Lidar_Manager()

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
                 dictObs=False, limited_time=False, display_lidar=False):
        
        # in radians
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,))   
        
        super().__init__(speed, obs_shape, conf, no_terminal, time_limit, dictObs, limited_time, display_lidar)
        ## TODO: fix this
        self.action_size = 2
        
    def get_action(self, action) -> np.ndarray:
        # if type(action) != int:
        #     return action.reshape(1, -1)
        steer = np.clip(action, a_min=-1, a_max=1)[0]
        # steer = np.clip(action, a_min=-1, a_max=1)
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


def test_env(debug=False):
    env = create_f110env(no_terminal=False, env_time_limit=0, render=True, continuous_action=True, display_lidar=True)
    policy = GapFollowPolicy()
    wp_manager = Waypoints_Manager(save_wp=False)

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
            target_idx = metric['target_idx']
            # print(action)
            obs, step_reward, done, info = env.step(action[0])
            if env.display_lidar:
                env.lidarManager.update_lidar_windows(wait=1, obs=obs, target_idx=target_idx)
            if i % 10 == 0:
                # print(f'step_reward: {step_reward}')
                # print(min(obs['vecobs']))
                # print(env.wpManager.get_lateral_error(obs['raw_obs']))
                # print(step_reward)
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