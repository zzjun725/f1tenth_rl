import fire
import torch
import os
import json
from config.get_rlConfig import get_rlf110_cfg, path_filler
from argparse import Namespace
from baselineAgents.ppo_continuous import PPO
from baselineAgents.ppo_agent import PPOAgent
from f110_rlenv import create_f110env


def policy(a, max_action):
    # from [0,1] to [-max,max]
    return 2 * (a - 0.5) * max_action


def evaluate_policy(env, model, render, steps_per_epoch, max_action):
    scores = 0
    turns = 2
    for j in range(turns):
        s, done, ep_r, steps = env.reset(), False, 0, 0
        while not (done or (steps >= steps_per_epoch)):
            # Take deterministic actions at test time
            a, logprob_a = model.evaluate(s)
            act = policy(a, max_action)  # [0,1] to [-max,max]
            s_prime, r, done, info = env.step(act)
            # r = Reward_adapter(r, EnvIdex)

            ep_r += r
            steps += 1
            s = s_prime
            if render:
                env.render()
        scores += ep_r
    return scores / turns


def evaluate_continuous_ppo():
    env_cfg = json.load(open(os.path.join(path_filler('config'), 'rlf110_env_cfg.json')))
    env_cfg['dictObs'] = False
    env_cfg['obs_shape'] = 54
    env = create_f110env(**env_cfg)

    cfg_path = os.path.join(path_filler('config'), 'rlf110_ppo_continuouscfg.json')
    kwargs = json.load(open(cfg_path))
    opt = Namespace(**kwargs)
    kwargs['state_dim'] = env.observation_space.shape[0]
    kwargs['action_dim'] = env.action_space.shape[0]
    kwargs['env_with_Dead'] = False
    model = PPO(**kwargs)
    # model.actor.load_state_dict(torch.load('./evaluate_model/ppo/ppo_actor70000.pth'))
    # model.critic.load_state_dict(torch.load('./evaluate_model/ppo/ppo_critic70000.pth'))
    model.load(episode=25000)

    for _ in range(5):
        evaluate_policy(env, model, True, 10000, 1)
        print('finish one episode')


def evaluate_discrete_ppo():
    cfg_path = os.path.join(path_filler('config'), f'rlf110_ppocfg.json')
    cfg = Namespace(**json.load(open(cfg_path)))
    env_cfg = json.load(open(os.path.join(path_filler('config'), 'rlf110_env_cfg.json')))
    env_cfg['speed'] = 2
    env_cfg['dictObs'] = False
    env_cfg['obs_shape'] = 27
    env_cfg['continuous_action'] = False
    env = create_f110env(**env_cfg)
    model = PPOAgent(env=env, lr_actor=0.001, lr_critic=0.001, gamma=0.99, K_epochs=3,
                     eps_clip=0.2, conf=cfg, continuous_action=False)
    model.load('ppo/bestmodel_step78000')
    model.eval(render=True)


if __name__ == '__main__':
    fire.Fire({
        'continuous': evaluate_continuous_ppo,
        'discrete': evaluate_discrete_ppo
    })


