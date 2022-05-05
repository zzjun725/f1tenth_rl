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

from utils import path_filler, render_callback
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import random
from math import pi
from common.policys import GapFollowPolicy, RandomPolicy
import cv2
from time import strftime, gmtime
import json
import os


def create_f110env(**kargs):
    # load simulator config file(define map, waypoints, etc)
    with open(kargs['sim_cfg_file']) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    sim_cfg = Namespace(**conf_dict)

    # choose continuous/discrete action space
    if kargs['lidar_action']:
        env = F110Env_LiDAR_Action(sim_cfg=sim_cfg, **kargs)
    elif kargs['continuous_action']:
        env = F110Env_Continuous_Action(sim_cfg=sim_cfg, **kargs)
    else:
        env = F110Env_Discrete_Action(sim_cfg=sim_cfg, **kargs)

    # render option
    if kargs['render_env']:
        print('render')
        env.f110.add_render_callback(render_callback)
    return env


def create_dictObs_eval_env(cfg=None):
    env_cfg = json.load(open(os.path.join(path_filler('config'), 'rlf110_env_cfg.json')))
    env_cfg['dictObs'] = True
    env_cfg['render_env'] = True
    env_cfg['obs_shape'] = 108
    env = create_f110env(**env_cfg)
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
        dist = np.linalg.norm(wp_xyaxis - cur_position.reshape(2, 1), axis=0)
        nearst_idx = np.argmin(dist)
        nearst_point = wp_xyaxis[:, nearst_idx]
        return nearst_point

    def get_wpbased_error(self, raw_obs, ego_idx=0):
        # pose: (x, y, yaw)
        x, y, theta = raw_obs['poses_x'][ego_idx], raw_obs['poses_y'][ego_idx], raw_obs['poses_theta'][ego_idx]
        pose = np.array([x, y, theta])
        nearstP = self.get_nearest_wp(pose[:2])

        euler_error = np.linalg.norm(nearstP-pose[:2])
        # return euler_error

        yaw = pose[2]
        local2global = np.array([[np.cos(yaw), -np.sin(yaw), 0, pose[0]],
                                 [np.sin(yaw), np.cos(yaw), 0, pose[1]],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]])

        # wp_xyaxis = self.wp[:2]
        global2local = np.linalg.inv(local2global)
        nearstP_local = global2local @ np.array([nearstP[0], nearstP[1], 0, 1])
        lateral_error = nearstP_local[1]
        return abs(lateral_error)

    def save_wp(self, raw_obs, ego_idx=0):
        x, y, theta = raw_obs['poses_x'][ego_idx], raw_obs['poses_y'][ego_idx], raw_obs['poses_theta'][ego_idx]
        self.log.write('%f, %f, %f\n' % (x, y, theta))


class Lidar_Manager:
    def __init__(self, scan_dim=108, window_H=250, scanScale=10, xy_range=30, display_lidar=False):
        """
        Include the code of displays the Lidar scan and reconstruction using cv2
        """
        self.display_lidar = display_lidar
        window_W = scan_dim + window_H
        self.obs_gap = int(1080/scan_dim)
        self.scan_dim = scan_dim
        self.dimension = scan_dim
        self.map_size = window_H
        self.map = np.zeros((self.map_size, self.map_size))

        self.lidar_dmin = 0
        self.lidar_dmax = 30
        self.angle_min = -135
        self.angle_max = 135
        self.x_min, self.x_max = -xy_range, xy_range
        self.y_min, self.y_max = -xy_range, xy_range
        self.resolution = (2*xy_range) / (window_H-1)
        print(f'lidar_resolution{self.resolution}')
        self.lidar_angles = np.linspace(self.angle_min, self.angle_max, self.dimension) * np.pi / 180
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
        x = distance * np.cos(self.lidar_angles)
        y = distance * np.sin(self.lidar_angles)
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
            fill_x = np.linspace(cell_indices[1, i], cell_indices[1, i + 1], endpoint=False, dtype='int')
            fill_y = np.linspace(cell_indices[0, i], cell_indices[0, i + 1], endpoint=False, dtype='int')
            self.map[fill_x, fill_y] = 1

    def update_scan2map(self, lidar_1d, color='r'):
        if color == 'r':
            color_tuple = (0, 0, 255)
        elif color == 'b':
            color_tuple = (255, 0, 0)

        self.map = np.zeros((self.map_size, self.map_size))
        self.lidar_reconstructPic = np.zeros((self.map_size, self.map_size, 3), np.uint8)

        self.distance = lidar_1d
        self.dimension = len(self.distance)

        x, y = self.rays2world(self.distance)
        cell_indices = self.grid_cell_from_xy(x, y)
        self.map[cell_indices[1, :], cell_indices[0, :]] = 1

        if self.interpolate_or_not:
            self.interpolate(cell_indices[:, :])

        cell_indices_line = np.vstack([cell_indices[0, :], cell_indices[1, :]]).T
        cv2.polylines(self.lidar_reconstructPic, [cell_indices_line], False, color_tuple, 2)
        return self.lidar_reconstructPic
        # import ipdb; ipdb.set_trace()

    # plt.imshow(self.map)
    # plt.show()

    def update_scan(self, best_p_idx=None, scan=None):
        self.lidar_scanPic = np.zeros((self.window_H, self.scan_dim, 3), np.uint8)
        scan = (scan * self.scanScale).astype(np.int64)
        scan = np.vstack([np.arange(len(scan)).astype(np.int64), self.window_H - scan]).T

        cv2.polylines(self.lidar_scanPic, [scan], False, (0, 0, 255), 2)
        if best_p_idx:
            target = scan[max(0, best_p_idx - 1):best_p_idx + 1, :]
            cv2.polylines(self.lidar_scanPic, [target], False, (0, 255, 0), 2)
        return self.lidar_scanPic

    def update_lidar_windows(self, wait=1, obs=None, target_idx=None):
        # if 'scans' in obs.keys():
        scan = obs['scans']
        scan = scan[::self.obs_gap]
        # scan = obs['scans'][0]
        self.update_scan2map(scan)
        self.update_scan(best_p_idx=target_idx, scan=scan)

        if self.display_lidar:
            self.lidar_window = np.hstack([self.lidar_scanPic, self.lidar_reconstructPic])
            cv2.imshow('debug', self.lidar_window)
            cv2.waitKey(wait)

class F110Env_RL:
    def __init__(self, continuous_action=True, sim_cfg=None, **kargs) -> None:

        for key, value in kargs.items():
            # print(key)
            setattr(self, key, value)

        self.f110 = F110Env(map=sim_cfg.map_path, map_ext=sim_cfg.map_ext, num_agents=1)
        self.conf = sim_cfg
        self.observation_space = spaces.Box(low=0, high=1000, shape=(self.obs_shape, 1))
        self.observation_gap = 1080 // self.obs_shape

        # waypoints Manager, for lateral error
        self.wpManager = Waypoints_Manager(sim_cfg.wpt_path, save_wp=False, load_wp=True)
        self.wpManager.load_wp()
        self.waypoints_xytheta = self.wpManager.wp.T
        self.lateral_error_thres = 0.2

        # lidar
        self.lidarManager = Lidar_Manager(scan_dim=self.obs_shape, display_lidar=self.display_lidar)

        # for offline data storage
        self.action_size = self.action_space.n if hasattr(self.action_space, 'n') else self.action_space.shape[0]
        # self._max_episode_steps = time_limit

        self.episode = []

    def reset(self):
        starting_idx = random.sample(range(len(self.waypoints_xytheta)), 1)
        # print(self.waypoints_xytheta[starting_idx])
        x, y = self.waypoints_xytheta[starting_idx][0, 0], self.waypoints_xytheta[starting_idx][
            0, 1]  # because self.waypoints_xytheta[starting_idx] has shape(1,3)
        # theta = 2*random.random() - 1
        theta_noise = (2*random.random() - 1) * 0.2
        theta = self.waypoints_xytheta[starting_idx][0, 2] + theta_noise
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
            obs['scans'] = raw_obs['scans'][0]
            # obs['raw_obs'] = raw_obs
            self.episode = [obs.copy()]

        # time limit
        self.step_ = 0

        return obs

    def get_obs(self, raw_obs: dict):
        obs = raw_obs['scans'][0][::self.observation_gap]
        # print(self.observation_gap)
        if self.dictObs:
            if len(obs.shape) == 1:
                return {'vecobs': obs}  # Vector env
            else:
                return {'image': obs}  # Image env
        else:
            return obs

    def get_reward(self, raw_obs: dict, crash: bool):
        # if 'raw_obs' in raw_obs.keys():
        #     raw_obs = raw_obs['raw_obs']
        wp_based_error = self.wpManager.get_wpbased_error(raw_obs)

        if crash:
            reward = -0.05
        elif wp_based_error > self.lateral_error_thres:
            # print(f'lateral_error:{wp_based_error}')
            reward = 0.0
            # reward -= wp_based_error * 0.01
        else:
            reward = 0.02
        return reward

    def render(self, mode='human'):
        self.f110.render(mode)

    def close(self):
        self.f110.close()

    def get_action(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError


class F110Env_Discrete_Action(F110Env_RL):
    def __init__(self, continuous_action=True, sim_cfg=None, **kargs):
        self.action_space = spaces.Discrete(3)
        super().__init__(continuous_action, sim_cfg, **kargs)
        # steer, speed
        for key, value in kargs.items():
            setattr(self, key, value)

        self.f110_action = np.array([
            # go straight
            [1, self.speed],  # go left
            [-1, self.speed],  # go right
            [0, self.speed],
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
                assert isinstance(action, np.ndarray) and action.shape == (
                self.action_size,), "Wrong one-hot action shape"
                action_vec = action

        action = self.get_action(action)
        raw_obs, reward, done, info = self.f110.step(action)
        info = {}
        obs = self.get_obs(raw_obs)
        reward = self.get_reward(raw_obs, done)

        # time_limit
        if self.limited_time:
            self.step_ += 1
            if self.step_ >= self.env_time_limit:
                done = True
                info['time_limit'] = True

        ## info
        if self.dictObs:
            obs['action'] = action_vec
            obs['reward'] = np.array(reward)
            obs['terminal'] = np.array(False if self.no_terminal else done)
            obs['reset'] = np.array(False)
            obs['scans'] = raw_obs['scans'][0]

            self.episode.append(obs.copy())
            if done:
                episode = {k: np.array([t[k] for t in self.episode]) for k in self.episode[0]}
                info['episode'] = episode

        return obs, reward, done, info
        # return obs, reward, done, info


########### Bowen Jiang, Added on Apr 17, 2022
class F110Env_Continuous_Action(F110Env_RL):
    def __init__(self, continuous_action=True, sim_cfg=None, **kargs):
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,))
        super().__init__(continuous_action, sim_cfg, **kargs)
        # steer, speed
        for key, value in kargs.items():
            setattr(self, key, value)
        self.action_size = self.action_space.shape[0]

    def get_action(self, action) -> np.ndarray:
        # if type(action) != int:
        #     return action.reshape(1, -1)
        # try:
        if type(action) == np.ndarray:
            action = action[0]
        # except:
        #     print('no valid steer')
        #     print(action)
        #     action = 0.0
        steer = np.clip(action, a_min=-1, a_max=1)
        # import ipdb; ipdb.set_trace()
        # steer = np.clip(action, a_min=-1, a_max=1)
        action = np.array([steer, self.speed]).astype(np.float32)
        return action.reshape(1, -1)

    def step(self, action):
        exe_action = self.get_action(action)
        # import ipdb
        # ipdb.set_trace()
        # print(action)
        # done = False
        raw_obs, reward, done, info = self.f110.step(exe_action)
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
            if self.step_ >= self.env_time_limit:
                done = True
                info['time_limit'] = True

        ## info
        if self.dictObs:
            obs['action'] = action
            obs['reward'] = np.array(reward)
            obs['terminal'] = np.array(False if self.no_terminal else done)
            obs['reset'] = np.array(False)
            obs['scans'] = raw_obs['scans'][0]
            # import ipdb
            # ipdb.set_trace()

            self.episode.append(obs.copy())
            if done:
                # import ipdb
                # ipdb.set_trace()
                # print(self.episode)
                episode = {k: np.array([t[k] for t in self.episode]) for k in self.episode[0]}
                info['episode'] = episode

        return obs, reward, done, info


class F110Env_LiDAR_Action(F110Env_RL):
    def __init__(self, continuous_action=True, sim_cfg=None, **kargs):
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,))
        super().__init__(continuous_action, sim_cfg, **kargs)
        # steer, speed
        for key, value in kargs.items():
            setattr(self, key, value)
        self.action_size = self.action_space.shape[0]

    def get_action(self, action) -> np.ndarray:
        # if type(action) != int:
        #     return action.reshape(1, -1)
        # try:
        if type(action) == np.ndarray:
            action = action[0]
        # except:
        #     print('no valid steer')
        #     print(action)
        #     action = 0.0
        steer = np.clip(action, a_min=-1, a_max=1)
        # import ipdb; ipdb.set_trace()
        # steer = np.clip(action, a_min=-1, a_max=1)
        action = np.array([steer, self.speed]).astype(np.float32)
        return action.reshape(1, -1)

    def step(self, action):
        exe_action = self.get_action(action)
        # import ipdb
        # ipdb.set_trace()
        # print(action)
        # done = False
        raw_obs, reward, done, info = self.f110.step(exe_action)
        info = {}
        ##  make 2 step with the same action
        # step = 3
        # while not done and step > 0:
        #     raw_obs, reward, done, info = self.f110.step(action)
        #     step -= 1
        ##  give penalty for hitting the wall
        obs = {}
        reward = self.get_reward(raw_obs, done)

        # time_limit
        if self.limited_time:
            self.step_ += 1
            if self.step_ >= self.env_time_limit:
                done = True
                info['time_limit'] = True

        ## info
        if self.dictObs:
            obs['action'] = action
            obs['reward'] = np.array(reward)
            obs['terminal'] = np.array(False if self.no_terminal else done)
            obs['reset'] = np.array(False)
            obs['scans'] = raw_obs['scans'][0]
            obs['image'] = cv2.resize(self.lidarManager.lidar_reconstructPic[:, :, 2], (25, 25))
            print((obs['image'] > 0).sum())

            cv2.imshow('debug', obs['image'])
            cv2.waitKey(1)
            # import ipdb
            # ipdb.set_trace()

            self.episode.append(obs.copy())
            if done:
                # import ipdb
                # ipdb.set_trace()
                # print(self.episode)
                episode = {k: np.array([t[k] for t in self.episode]) for k in self.episode[0]}
                info['episode'] = episode

            self.lidarManager.update_lidar_windows(wait=1, obs=obs)

        return obs, reward, done, info

    def reset(self):
        starting_idx = random.sample(range(len(self.waypoints_xytheta)), 1)
        # print(self.waypoints_xytheta[starting_idx])
        x, y = self.waypoints_xytheta[starting_idx][0, 0], self.waypoints_xytheta[starting_idx][
            0, 1]  # because self.waypoints_xytheta[starting_idx] has shape(1,3)
        # theta = 2*random.random() - 1
        theta_noise = (2*random.random() - 1) * 0.2
        theta = self.waypoints_xytheta[starting_idx][0, 2] + theta_noise
        starting_pos = np.array([[x, y, theta]])
        # starting_pos[-1] += 0.5
        raw_obs, _, _, _ = self.f110.reset(starting_pos)
        # raw_obs, _, _, _ = self.f110.reset(np.array([[self.conf.sx, self.conf.sy, self.conf.stheta]]))
        obs = {}

        # dict
        if self.dictObs:
            obs['action'] = np.zeros(self.action_size)
            obs['reward'] = np.array(0.0)
            obs['terminal'] = np.array(False)
            obs['reset'] = np.array(True)
            obs['scans'] = raw_obs['scans'][0]
            obs['image'] = cv2.resize(self.lidarManager.lidar_reconstructPic[:, :, 2], (25, 25))
            self.episode = [obs.copy()]

        # time limit
        self.step_ = 0
        return obs


def test_env(debug=False):
    env_cfg = json.load(open(os.path.join(path_filler('config'), 'rlf110_env_cfg.json')))
    env_cfg['render_env'] = True
    env_cfg['display_lidar'] = True
    env_cfg['obs_shape'] = 1080
    env_cfg['lidar_action'] = True
    env_cfg['sim_cfg_file'] = "/home/mlab/zhijunz/dreamerv2_dev/f1tenth_rl/examples/RL_example/config/maps/config_example_map.yaml"
    # env_cfg['lidar_action'] = True
    # import ipdb; ipdb.set_trace()
    env = create_f110env(**env_cfg)
    policy = GapFollowPolicy()
    # policy = RandomPolicy(action_space=env.action_space)
    wp_manager = Waypoints_Manager(save_wp=False)

    for ep_i in range(5):
        obs = env.reset()
        done = False
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

            ##### use policy ######
            action, metric = policy(obs)
            # target_idx = metric['target_idx']
            # print(action)
            obs, step_reward, done, info = env.step(action)
            # print(obs['reward'])
            ##### random #######
            # obs, step_reward, done, info = env.step(0)
            # env.lidarManager.update_lidar_windows(wait=1, obs=obs)
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
