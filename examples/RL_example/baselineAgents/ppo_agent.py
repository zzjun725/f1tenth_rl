# -- coding: utf-8 --
import os
import glob
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, MultivariateNormal
from torch.distributions import Categorical
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import random
from torch.nn.utils import clip_grad_norm
import gym
from tqdm import tqdm


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
    def __init__(self, state_dim, action_dim, continuous_action=False, device=None, action_var_init=0.1):
        super(Actor, self).__init__()
        self.continuous_action = continuous_action
        self.continuous_action_scale = 2.0
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, action_dim),
        )
        
        if self.continuous_action:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_var_init).to(device)
            # self.action_var = torch.tensor([action_var_init]).to(device)
        self.device = device

    def forward(self, x):
        x = self.net(x)
        if self.continuous_action:
            action_mean = torch.tanh(x)
            return action_mean * self.continuous_action_scale
        else:
            action_prob = F.softmax(x, dim=-1)
            return action_prob

    def act(self, x):
        if self.continuous_action:
            action_mean = self.forward(x)
            action_var = self.action_var.expand_as(action_mean)
            dist = Normal(action_mean, action_var)
            action = dist.sample()
            action_logprob = dist.log_prob(action.reshape(-1, self.action_dim))
            # action_logits = None
        else:
            action_probs = self.forward(x)
            # try:
            dist = Categorical(action_probs)
            # except:
            #     import ipdb
            #     ipdb.set_trace()
            action = dist.sample()
            action_logprob = dist.log_prob(action.reshape(-1, 1))
            # action_logits = action_probs

        # TODO: Check detach
        return action, action_logprob

    def set_action_std(self, new_action_var):
        if self.continuous_action:
            self.action_var = torch.full((self.action_dim,), new_action_var).to(self.device)
    
    def decay_action_std(self, action_std_decay_rate, min_action_std):
        if self.continuous_action:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)


class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        value = self.net(x)
        return value


class PPOAgent:
    def __init__(self, env,
                 lr_actor, lr_critic,
                 gamma=0.99, K_epochs=3, eps_clip=0.2, gae_lambda=0.95,
                 conf=None, continuous_action=False):
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

        self.continuous_action = continuous_action
        state_dim = env.observation_space.shape[0]
        # import ipdb
        # ipdb.set_trace()
        if continuous_action:
            action_dim = env.action_space.shape[0]
        else:
            action_dim = env.action_space.n
        self.action_dim = action_dim
        self.actor = Actor(state_dim, action_dim, continuous_action, device=device).to(device)
        self.critic = Critic(state_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.actor_old = Actor(state_dim, action_dim, continuous_action, device=device).to(device)
        self.actor_old.load_state_dict(self.actor.state_dict())

    def put_data(self, *transtion):
        self.buffer.put_data(*transtion)

    def clip_grad(self):
        for p in self.actor.parameters():
            clip_grad_norm(p, 10)
        for p in self.critic.parameters():
            clip_grad_norm(p, 10)

    def update(self):
        s, a, r, s_next, logprob_a, done_mask = self.buffer.get_batch()
        s, a = torch.tensor(np.array(s), dtype=torch.float).to(self.device), torch.tensor(np.array(a)).to(self.device)
        r, s_next = torch.tensor(np.array(r)).to(self.device), torch.tensor(np.array(s_next), dtype=torch.float).to(self.device)
        logprob_a, done_mask = torch.tensor(np.array(logprob_a)).to(self.device), torch.tensor(np.array(done_mask)).to(self.device)
        actor_loss_sum = 0
        critic_loss_sum = 0
        # import ipdb
        # ipdb.set_trace()

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
            if advantage.shape[1] > 1:
                advantage = (advantage - advantage.mean()) / ((advantage.std()+1e-4))
            if self.continuous_action:
                action_mean = self.actor(s)
                action_var = self.actor.action_var.expand_as(action_mean)
                dist = Normal(action_mean, action_var)
                pi_a = dist.log_prob(a.reshape(-1, self.action_dim))
                ratio = torch.exp(pi_a - logprob_a)  # a/b == exp(log(a)-log(b))
            else:
                pi = self.actor(s)
                pi_a = pi.gather(1, a)
                ratio = torch.exp(torch.log(pi_a) - logprob_a)  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage

            critic_loss = F.smooth_l1_loss(self.critic(s), td_target.detach())
            actor_loss = -torch.min(surr1, surr2).mean()
            # print('actor_loss', actor_loss)
            loss = actor_loss + critic_loss

            # try: 
            #     assert(loss == loss)
            # except:
            #     import ipdb
            #     ipdb.set_trace()
            
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
        for n_epi in tqdm(range(episode_num)):
            s = self.env.reset()
            self.env.render()
            done = False
            score = 0
            while not done:
                for i in range(update_interval):
                    total_timestep += 1
                    a, logprob = self.actor_old.act(torch.tensor(np.array([s]), dtype=torch.float).to(self.device))
                    a = a.cpu().item()
                    if self.continuous_action:
                        s_next, r, done, info = self.env.step(np.array([a]))
                        s_next = s_next.squeeze()  # in case s not 1-dimension                        
                    else:
                        s_next, r, done, info = self.env.step(a)
                    # print(a)

                    self.put_data(*(s, a, r, s_next, logprob[0].item(), done))
                    s = s_next
                    score += r
                    if total_timestep % render_interval == 0:
                        print("# of episode :{}, last 10 ep avg score : {:.1f}".format(n_epi, np.mean(scores[-10:])))
                        print('scores', np.mean(scores), 'actor_losses', np.mean(actor_losses), 'critic_losses', np.mean(critic_losses))
                        # self._plot(total_timestep, scores, actor_losses, critic_losses)
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
                    # if self.continuous_action:
                    a, logprob = self.actor_old.act(torch.tensor(np.array([s]), dtype=torch.float).to(self.device))
                    # prob = self.actor_old.act(torch.tensor(np.array([s]), dtype=torch.float).to(self.device))
                    # m = Categorical(prob)
                    # a = m.sample().item()
                    # a = torch.argmax(prob).item()
                    if self.continuous_action:
                        a = a.cpu().numpy()
                    else:
                        a = a.cpu().item()
                    s_next, r, done, info = self.env.step(a)
                    s_next = s_next.squeeze()  # in case s not 1-dimension
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


def main_discrete():
    # env = gym.make('MountainCar-v0').unwrapped
    env = gym.make('CartPole-v0')
    model = PPOAgent(env=env, lr_actor=0.001, lr_critic=0.001, gamma=0.99, K_epochs=3,
                     eps_clip=0.1, continuous_action=False)
    score = 0.0
    print_interval = 20
    T_horizon = 20
    episode_num = 1000

    model.train(episode_num=episode_num, update_interval=20, render_interval=2000, save_interval=3000)
    env.close()


def main_continuous():
    # env = gym.make('MountainCar-v0').unwrapped
    # env = gym.make('CartPole-v0')
    env = gym.make('Pendulum-v0')
    model = PPOAgent(env=env, lr_actor=0.001, lr_critic=0.001, gamma=0.99, K_epochs=3,
                     eps_clip=0.1, continuous_action=True)
    score = 0.0
    print_interval = 20
    T_horizon = 20
    episode_num = 1000

    model.train(episode_num=episode_num, update_interval=20, render_interval=2000, save_interval=3000)
    env.close()


if __name__ == '__main__':
    main_continuous()

    ### test actor
    
    # device = torch.device('cuda:0')
    # actor = Actor(state_dim=2, action_dim=2, device=device, action_var_init=0.1, continuous_action=True).to(device)
    # fake_s = torch.ones((2)).to(device)
    # a, logprob, _ = actor.act(fake_s)

