# -- coding: utf-8 --
from typing import Dict, List, Tuple

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import collections
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from segment_tree import MinSegmentTree, SumSegmentTree
from torch.nn.utils import clip_grad_norm_
from IPython.display import clear_output
import os
import json
from argparse import Namespace


class Net(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        """Initialization."""
        super(Net, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        return self.layers(x)


class DuelNet(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        """Initialization."""
        super(DuelNet, self).__init__()

        # set common feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
        )

        # set advantage layer
        self.advantage_layer = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        )

        # set value layer
        self.value_layer = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        feature = self.feature_layer(x)
        value = self.value_layer(feature)
        advantage = self.advantage_layer(feature)
        q = value + advantage - advantage.mean(dim=-1, keepdim=True)

        return q


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


class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized Replay buffer.

    Attributes:
        max_priority (float): max priority
        tree_ptr (int): next index of tree
        alpha (float): alpha parameter for prioritized replay buffer
        sum_tree (SumSegmentTree): sum tree for prior
        min_tree (MinSegmentTree): min tree for min prior to get max weight

    """

    def __init__(
            self,
            obs_dim: int,
            size: int,
            batch_size: int = 32,
            alpha: float = 0.6
    ):
        """Initialization."""
        assert alpha >= 0

        super(PrioritizedReplayBuffer, self).__init__(obs_dim, size, batch_size)
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha

        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def store(
            self,
            obs: np.ndarray,
            act: int,
            rew: float,
            next_obs: np.ndarray,
            done: bool
    ):
        """Store experience and priority."""
        super().store(obs, act, rew, next_obs, done)

        self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.max_size

    def sample_batch(self, beta: float = 0.4) -> Dict[str, np.ndarray]:
        """Sample a batch of experiences."""
        assert len(self) >= self.batch_size
        assert beta > 0

        indices = self._sample_proportional()

        obs = self.obs_buf[indices]
        next_obs = self.next_obs_buf[indices]
        acts = self.acts_buf[indices]
        rews = self.rews_buf[indices]
        done = self.done_buf[indices]
        weights = np.array([self._calculate_weight(i, beta) for i in indices])

        return dict(
            obs=obs,
            next_obs=next_obs,
            acts=acts,
            rews=rews,
            done=done,
            weights=weights,
            indices=indices,
        )

    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)

    def _sample_proportional(self) -> List[int]:
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / self.batch_size

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)

        return indices

    def _calculate_weight(self, idx: int, beta: float):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)

        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight

        return weight


class D3QNAgent:

    def __init__(
            self,
            env,
            memory_size: int,
            batch_size: int,
            target_update: int,
            epsilon_decay: float,
            max_epsilon: float = 0.8,
            min_epsilon: float = 0.1,
            gamma: float = 0.99,
            beta: float = 0.6,
            prior_eps: float = 1e-6,
            conf=None
    ):
        self.config = conf

        # obs_dim = env.observation_space.shape[0]
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        self.env = env
        self.memory = ReplayBuffer(obs_dim, memory_size, batch_size)
        self.batch_size = batch_size
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.reduc_epsilon = 0.05
        self.target_update = target_update
        self.gamma = gamma
        self.beta = beta
        self.prior_eps = prior_eps

        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(self.device)

        self.dqn = DuelNet(obs_dim, action_dim).to(self.device)
        self.dqn_target = DuelNet(obs_dim, action_dim).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        self.optimizer = optim.Adam(self.dqn.parameters())
        self.transition = list()
        self.is_test = False

    def select_action(self, obs: np.ndarray) -> int:
        # obs = self.env.get_obs(obs)
        if self.epsilon > np.random.random():
            action = self.env.action_space.sample()
        else:
            action = self.dqn(
                torch.FloatTensor(obs).to(self.device)
            ).argmax()
            action = action.detach().cpu().numpy()
        # print(action)
        self.transition = [obs, action]
        return action

    def step(self, action: int) -> Tuple[np.ndarray, np.float64, bool]:
        next_obs, reward, done, info = self.env.step(action)
        self.transition += [reward, next_obs, done]
        self.memory.store(*self.transition)
        return next_obs, reward, done

    def eval_step(self, obs: np.ndarray):
        action = self.dqn(
            torch.FloatTensor(obs).to(self.device)
        ).argmax()
        action = action.detach().cpu().numpy()
        next_obs, reward, done, info = self.env.step(action)
        return next_obs, reward, done

    def update_epsilons(self, scores):
        # if np.mean(scores[-10:]) < np.mean(scores[-20:-10]):
        #     self.epsilon = min(self.epsilon+self.reduc_epsilon, self.max_epsilon)
        # else:
        #     self.epsilon = max(self.epsilon-self.reduc_epsilon, self.min_epsilon)
        self.epsilon = max(self.epsilon-self.epsilon_decay, self.min_epsilon)

    def save(self, filename):
        if self.config:
            model_path = self.config.model_dir
            torch.save(self.dqn.state_dict(), model_path + f'/{filename}.pkl')

    def load(self, filename):
        model_path = self.config.model_dir
        state_dict = torch.load(model_path + f'/{filename}', map_location=lambda storage, loc: storage)
        self.dqn.load_state_dict(state_dict)

    def eval(self, reload_model=False, render=False, model=None):
        self.is_test = True
        if reload_model:
            if model:
                self.load(filename=model)
            else:
                for path in os.listdir(self.config.model_dir)[::-1]:
                    if path.startswith('best'):
                        self.load(filename=path)
                        break
        obs = self.env.reset()
        done = False
        score = 0
        while not done:
            next_obs, reward, done = self.eval_step(obs)
            obs = next_obs
            score += reward
            if render:
                self.env.render()
        self.env.close()
        return score

    def train(self, num_steps: int, render_interval: int = 200, save_interval: int = 1000):
        """Train the agent."""
        self.is_test = False
        obs = self.env.reset()
        update_cnt = 0
        epsilons = []
        losses = []
        scores = []
        score = 0
        best_score = -np.inf

        for step_idx in range(1, num_steps + 1):
            # obs = self.env.get_obs(obs)
            action = self.select_action(obs)
            next_obs, reward, done = self.step(action)
            obs = next_obs
            score += reward

            # update beta
            # fraction = min(step_idx / num_steps, 1.0)
            # self.beta = self.beta + fraction * (1.0 - self.beta)

            # if episode ends
            if done:
                scores.append(score)
                score = 0
                if len(scores) > 20:
                    self.update_epsilons(scores)
                epsilons.append(self.epsilon)
                obs = self.env.reset()

            if len(self.memory) >= self.batch_size:
                # samples = self.memory.sample_batch(self.beta)
                samples = self.memory.sample_batch()
                # PER
                # weights = torch.FloatTensor(
                #     samples["weights"].reshape(-1, 1)
                # ).to(self.device)
                # indices = samples["indices"]
                # elementwise_loss = self._compute_dqn_loss(samples, elementwise=True)
                # loss = torch.mean(elementwise_loss * weights)
                # PER

                ## LOSS and clip grad
                loss = self._compute_dqn_loss(samples, elementwise=False)
                self.optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.dqn.parameters(), 10.0)
                self.optimizer.step()
                losses.append(loss.item())
                update_cnt += 1

                # PER: update priorities
                # loss_for_prior = elementwise_loss.detach().cpu().numpy()
                # new_priorities = loss_for_prior + self.prior_eps
                # self.memory.update_priorities(indices, new_priorities)

                ## update target Q-net
                if update_cnt % self.target_update == 0:
                    self._target_hard_update()

            ## save_model
            if step_idx % save_interval == 0 and step_idx > 1:
                self.save(f'step{step_idx}')
                cur_score = self.eval(reload_model=False, render=False)
                if cur_score >= best_score or cur_score >= 8:
                    self.save(f'bestmodel_step{step_idx}')
                    best_score = cur_score
                self.is_test = False

            ## plotting
            if step_idx % render_interval == 0 and step_idx > 1:
                self._plot(step_idx, scores, losses, epsilons)
                self.test(render=True)
                print('episode', len(scores), '  steps', step_idx)
                print('current_score', np.mean(scores[-10:]))
                # self.is_test = False

        self.env.close()

    def test(self, render_times=5, render=False):
        """Test the agent."""
        self.is_test = True
        scores = []
        for i in range(render_times):
            obs = self.env.reset()
            done = False
            score = 0
            while not done:
                action = self.select_action(obs)
                next_obs, reward, done = self.step(action)
                obs = next_obs
                score += reward
                if render:
                    self.env.render()
            scores.append(score)

        print('end_test')
        print('test_scores', np.round(np.array(scores), 4))
        self.env.close()

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray], elementwise=False) -> torch.Tensor:
        device = self.device
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        curr_q_value = self.dqn(state).gather(1, action)
        next_q_value = self.dqn_target(next_state).gather(  # Double DQN
            1, self.dqn(next_state).argmax(dim=1, keepdim=True)
        ).detach()
        mask = 1 - done
        target = (reward + self.gamma * next_q_value * mask).to(self.device)

        # calculate dqn loss
        if elementwise:
            loss = F.smooth_l1_loss(curr_q_value, target, reduction='none')
        else:
            loss = F.smooth_l1_loss(curr_q_value, target)

        return loss

    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())

    def _plot(
            self,
            frame_idx: int,
            scores: List[float],
            losses: List[float],
            epsilons: List[float],
    ):
        """Plot the training progresses."""
        clear_output(True)
        plt.figure(figsize=(10, 24))
        plt.subplot(311)
        plt.title('frame %s. score: %s' % (frame_idx, np.mean(scores[-10:])))
        plt.plot(scores)
        plt.subplot(312)
        plt.title('frame %s. loss: %s' % (frame_idx, np.mean(losses)))
        plt.plot(losses)
        plt.subplot(313)
        plt.title('epsilons')
        plt.plot(epsilons)
        plt.show()


if __name__ == '__main__':
    # test agent
    # env_id = "MountainCar-v0"
    env_id = "CartPole-v0"
    env = gym.make(env_id)
    seed = 777

    def seed_torch(seed):
        torch.manual_seed(seed)
        if torch.backends.cudnn.enabled:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


    np.random.seed(seed)
    seed_torch(seed)
    env.seed(seed)
    # parameters
    num_frames = 200000
    memory_size = 1000
    batch_size = 32
    target_update = 100  # update in 100 episode
    epsilon_decay = 1 / 100

    agent = D3QNAgent(env, memory_size, batch_size, target_update, epsilon_decay, conf=None)
    agent.train(num_frames, render_interval=1000)
