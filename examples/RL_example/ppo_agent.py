# -- coding: utf-8 --
import os
import glob
import time
from datetime import datetime

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


class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.logprobs = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.next_states[:]

    def put_data(self, *transition):
        s, a, r, s_next, prob_a, done = transition
        self.states.append(s)
        self.rewards.append([r])
        self.actions.append([a])
        self.next_states.append(s_next)
        self.logprobs.append([prob_a])
        self.is_terminals.append([0] if done else [1])

    def get_batch(self):
        return self.states, self.actions, self.rewards, \
               self.next_states, self.logprobs, self.is_terminals


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        x = self.net(x)
        action_prob = F.softmax(x, dim=1)
        return action_prob


class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        value = self.net(x)
        return value


class PPOAgent:
    def __init__(self, env,
                 lr_actor, lr_critic,
                 gamma=0.99, K_epochs=3, eps_clip=0.2, gae_lambda=0.95,
                 conf=None):
        self.config = conf
        self.env = env
        device = torch.device('cpu')
        if (torch.cuda.is_available()):
            device = torch.device('cuda:0')
            torch.cuda.empty_cache()
            print("Device set to : " + str(torch.cuda.get_device_name(device)))
        else:
            print("Device set to : cpu")
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.actor_old = Actor(state_dim, action_dim).to(device)
        self.actor_old.load_state_dict(self.actor.state_dict())

    def put_data(self, *transtion):
        self.buffer.put_data(*transtion)

    def clip_grad(self):
        for p in self.actor.parameters():
            clip_grad_norm(p, 10)
        for p in self.critic.parameters():
            clip_grad_norm(p, 10)

    def update(self):
        s, a, r, s_next, prob_a, done_mask = self.buffer.get_batch()
        s, a = torch.tensor(s, dtype=torch.float).to(self.device), torch.tensor(a).to(self.device)
        r, s_next = torch.tensor(r).to(self.device), torch.tensor(s_next, dtype=torch.float).to(self.device)
        prob_a, done_mask = torch.tensor(prob_a).to(self.device), torch.tensor(done_mask).to(self.device)
        actor_loss_sum = 0
        critic_loss_sum = 0
        for i in range(self.K_epochs):
            # use td_target instead of MC to update the critic net
            td_target = r + self.gamma * self.critic(s_next) * done_mask
            delta = (td_target - self.critic(s)).cpu().detach().numpy()
            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = self.gamma * self.gae_lambda*advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float).to(self.device)

            pi = self.actor(s)
            pi_a = pi.gather(1, a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage

            critic_loss = F.smooth_l1_loss(self.critic(s), td_target.detach())
            actor_loss = -torch.min(surr1, surr2).mean()
            loss = actor_loss + critic_loss
            actor_loss_sum += actor_loss.item()
            critic_loss_sum += critic_loss.item()
            # check for anomaly
            # if abs(actor_loss.item()) > 10:
            #     print(ratio, pi_a, pi, surr1, surr2)

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

        self.actor_old.load_state_dict(self.actor.state_dict())
        self.buffer.clear()
        return actor_loss_sum/self.K_epochs, critic_loss_sum/self.K_epochs

    def train(self, episode_num, update_interval, render_interval, save_interval, eval_episode=5):
        total_timestep = 0
        scores = []
        actor_losses = []
        critic_losses = []
        best_score = -np.inf
        for n_epi in range(episode_num):
            s = self.env.reset()
            self.env.render()
            done = False
            score = 0
            while not done:
                for i in range(update_interval):
                    total_timestep += 1
                    prob = self.actor_old(torch.tensor([s], dtype=torch.float).to(self.device))
                    # print(prob)
                    m = Categorical(prob)
                    a = m.sample().item()
                    s_next, r, done, info = self.env.step(a)
                    self.env.render()
                    self.put_data(*(s, a, r/100, s_next, prob[0][a].item(), done))
                    s = s_next
                    score += r
                    if total_timestep % render_interval == 0:
                        print("# of episode :{}, last 10 ep avg score : {:.1f}".format(n_epi, np.mean(scores[-10:])))
                        self._plot(total_timestep, scores, actor_losses, critic_losses)
                        cur_score = self.eval(reload_model=False, render=True, episode_num=eval_episode)
                        print(f'test for {eval_episode} episodes, mean score{cur_score}')
                    if total_timestep % save_interval == 0:
                        self.save(f'step{total_timestep}')
                        cur_score = self.eval(reload_model=False, render=False, episode_num=eval_episode)
                        if cur_score >= best_score or cur_score > 10:
                            self.save(f'bestmodel_step{total_timestep}')
                            best_score = cur_score
                    if done:
                        scores.append(score)
                        break
                actor_loss, critic_loss = self.update()
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)

    def eval(self, reload_model=False, render=False, episode_num=5, model=None):
        scores = []
        for n_epi in range(episode_num):
            s = self.env.reset()
            done = False
            score = 0
            while not done:
                if render:
                    self.env.render()
                with torch.no_grad():
                    prob = self.actor_old(torch.tensor([s], dtype=torch.float).to(self.device))
                    # m = Categorical(prob)
                    # a = m.sample().item()
                    a = torch.argmax(prob).item()
                    s_next, r, done, info = self.env.step(a)
                    score += r
                    s = s_next
        scores.append(score)
        self.env.close()
        return np.mean(scores)


    def _plot(self, total_timestep, scores, actor_losses, critic_losses):
        plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.title('frame %s. score: %s' % (total_timestep, np.mean(scores[-10:])))
        plt.plot(scores)
        plt.subplot(132)
        plt.title('frame %s. actor_loss: %s' % (total_timestep, np.mean(actor_losses[-10:])))
        plt.plot(actor_losses)
        plt.subplot(133)
        plt.title('frame %s. critic_loss: %s' % (total_timestep, np.mean(actor_losses[-10:])))
        plt.plot(critic_losses)
        plt.show()

    def save(self, filename):
        if self.config:
            model_path = self.config.model_dir
            torch.save(self.actor_old.state_dict(), model_path + f'/{filename}_actor.pkl')
            torch.save(self.critic.state_dict(), model_path + f'/{filename}_critic.pkl')

    def load(self, filename):
        model_path = self.config.model_dir
        state_dict_actor = torch.load(model_path + f'/{filename}_actor.pkl', map_location=lambda storage, loc: storage)
        self.actor_old.load_state_dict(state_dict_actor)
        self.actor.load_state_dict(state_dict_actor)
        state_dict_critic = torch.load(model_path + f'/{filename}_critic.pkl', map_location=lambda storage, loc: storage)
        self.critic.load_state_dict(state_dict_critic)


def main():
    # env = gym.make('MountainCar-v0').unwrapped
    env = gym.make('CartPole-v0')
    model = PPOAgent(env=env,lr_actor=0.001, lr_critic=0.001, gamma=0.99, K_epochs=3,
                     eps_clip=0.1)
    score = 0.0
    print_interval = 20
    T_horizon = 20
    episode_num = 1000

    model.train(episode_num=episode_num, update_interval=20, render_interval=2000, save_interval=3000)

    for n_epi in range(10000):
        s = env.reset()
        done = False
        i = 0
        while not done:
            for t in range(T_horizon):
                i += 1
                prob = model.actor_old(torch.tensor([s], dtype=torch.float).to(model.device))
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, info = env.step(a)
                model.put_data(*(s, a, r/100, s_prime, prob[0][a].item(), done))
                s = s_prime
                score += r
                # if done or i == 20000:
                if done:
                    # print(f'done_{i}_{r}')
                    # done = True
                    # i = 0
                    break

            model.update()

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score / print_interval))
            # if score / print_interval > -1000:
            #     env = gym.make('MountainCar-v0')
            score = 0.0

    env.close()


if __name__ == '__main__':
    main()
