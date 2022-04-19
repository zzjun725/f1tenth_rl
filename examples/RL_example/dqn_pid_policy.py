# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 23:32:51 2022

@author: Zhiyang Chen
"""
#-----IMPORT PACKAGES
import yaml
import time
import numpy as np
from typing import Dict
from argparse import Namespace
from utils import render_callback, path_filler
from f110_rlenv import F110Env_Discrete_Action


#-----DISCRETE PID POLICY CLASS
class DQN_PID:
    """This is a pid policy for the example waypoints trajectory"""
    def __init__(self, action_space=[0, 1, 2], traj_dir = '/Users/gurub/Desktop/Learning in Robotics/Final_Project/ZZJ_git/f1tenth_rl-main/examples/example_waypoints.csv'):
        # define action space
        self.action_space = action_space # 0 is turn left, 1 is turn right, 2 is go straight
        
        # load trajectory
        self.traj = np.loadtxt(traj_dir, comments='#', delimiter=';')[:, [1, 2]] # only take the x and y positions
        
    
    def pid_reward(self, obs:dict) -> int:
        # get the position and heading of the agent
        pos, theta = np.array([obs['poses_x'][0], obs['poses_y'][0]]), obs['poses_theta'][0]
        
        # parameters
        lim_val, scale = np.sin(10*np.pi/180), 2.5
        
        # find p1 and p2
        nbr_idx = np.argmin(np.linalg.norm(self.traj-pos, axis=1))
        p1 = self.traj[nbr_idx]
        p2 = self.traj[0, :] if nbr_idx == len(self.traj)-1 else self.traj[nbr_idx+1, :]
        
        # compute norm of desired direction
        d = (p2-p1)/np.linalg.norm(p2-p1)
        d_des = (p1-pos) + scale*d
        d_des_norm = d_des/np.linalg.norm(d_des)
        
        # tabulate car direction
        car_dir = np.array([np.cos(theta), np.sin(theta)])
        
        # make decision about control
        error_vec = np.cross(d_des_norm, car_dir)
        
        # compute the reward 
        forward_reward, follow_reward = np.dot(car_dir, d_des_norm), np.linalg.norm(error_vec)
        if forward_reward > 0:
            follow_reward = -follow_reward if follow_reward > lim_val else follow_reward
        else:
            follow_reward *= -1
        
        pid_reward = 0.05*forward_reward + 0.05*follow_reward
        
        return pid_reward
    
    def __call__(self, obs:dict) -> int:
        # get the position and heading of the agent
        pos, theta = np.array([obs['poses_x'][0], obs['poses_y'][0]]), obs['poses_theta'][0]
        
        # parameters
        lim_val, scale = np.sin(10*np.pi/180), 2.5
        
        # find p1 and p2
        nbr_idx = np.argmin(np.linalg.norm(self.traj-pos, axis=1))
        p1 = self.traj[nbr_idx]
        p2 = self.traj[0, :] if nbr_idx == len(self.traj)-1 else self.traj[nbr_idx+1, :]
        
        # compute norm of desired direction
        d = (p2-p1)/np.linalg.norm(p2-p1)
        d_des = (p1-pos) + scale*d
        d_des_norm = d_des/np.linalg.norm(d_des)
        
        # tabulate car direction
        car_dir = np.array([np.cos(theta), np.sin(theta)])
        
        # make decision about control
        error_vec = np.cross(d_des_norm, car_dir)
        
        if np.dot(car_dir, d_des_norm) >= 0:
            if error_vec > lim_val:
                return self.action_space[1]
            elif error_vec < -lim_val:
                return self.action_space[0]
            else:
                return self.action_space[2]
        else:
            if error_vec > 0:
                return self.action_space[1]
            elif error_vec < 0:
                return self.action_space[0]
            else:
                return np.random.choice(self.action_space[:2])

#-----GENERATE SAMPLE TRAINING DATA
def get_train_data(iteration=3000, batch_size=50):
    # initialize environment
    with open('./config_example_map.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)
    env = F110Env_Discrete_Action(conf=conf)
    env.f110.add_render_callback(render_callback)
    obs_dim = env.observation_space.shape[0]
    
    rb = ReplayBuffer(obs_dim, iteration, batch_size) # initialize replay buffer
    pid = DQN_PID() # initialize controller
    
    # loop to collect trajectory
    obs = env.reset_full()
    for ite in range(iteration):
        action = pid(obs) if np.random.sample() < 0.2 else np.random.choice([0, 1, 2])
        next_obs, reward, done, info = env.step_full(action)
        rb.store(obs['scans'][0][::40], action, reward, next_obs['scans'][0][::40], done)
        obs = next_obs
        if done:
            obs = env.reset_full()  
            #env.close()
            #break
        #env.render()
    
    env.close()
        
    return rb

#-----REPLAY BUFFER
class ReplayBuffer:
    def __init__(self, obs_dim: int, size: int, batch_size: int = 32):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0

    def store(
            self,
            obs: np.ndarray,
            act: np.ndarray,
            rew: float,
            next_obs: np.ndarray,
            done: bool,
    ):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(obs=self.obs_buf[idxs],
                    next_obs=self.next_obs_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

    def __len__(self) -> int:
        return self.size

  
        



