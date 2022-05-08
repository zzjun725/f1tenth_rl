# -- coding: utf-8 --
import json
import os
from cv2 import INTER_MAX
import yaml
from argparse import Namespace
from utils import render_callback, path_filler
from f110_rlenv import F110Env_Discrete_Action, F110Env_Continuous_Action, create_f110env
from baselineAgents.ppo_agent import PPOAgent
from baselineAgents.ppo_continuous import PPO
from torch.utils.tensorboard import SummaryWriter
import os, shutil
from datetime import datetime
import argparse
from config.get_rlConfig import get_rlf110_cfg, path_filler
import json
from argparse import Namespace

'''Env Setting'''
# with open('./config_example_map.yaml') as file:
#     conf_dict = yaml.load(file, Loader=yaml.FullLoader)
# conf = Namespace(**conf_dict)
#
# continuous_action = True
# if continuous_action:
#     env = F110Env_Continuous_Action(sim_cfg=conf, dictObs=False)
#     eval_env = F110Env_Continuous_Action(sim_cfg=conf, dictObs=False)
# else:
#     env = F110Env_Discrete_Action(sim_cfg=conf, dictObs=False)
#     eval_env = F110Env_Continuous_Action(sim_cfg=conf, dictObs=False)
# print('continuous_action', continuous_action)
# env.f110.add_render_callback(render_callback)

env_cfg = json.load(open(os.path.join(path_filler('config'), 'rlf110_env_cfg.json')))
env_cfg['dictObs'] = False
env_cfg['obs_shape'] = 1080
env_cfg['env_time_limit'] = 5000
env_cfg['no_terminal'] = False
# env_cfg["sim_cfg_file"]= "/home/mlab/zhijunz/dreamerv2_dev/f1tenth_rl/examples/RL_example/config/maps/config_example_map.yaml"
env = create_f110env(**env_cfg)
eval_env = create_f110env(**env_cfg)

# train for discrete
# task = 'ppo'
# cfg_path = os.path.join(path_filler('config'), f'rlf110_{task}cfg.json')
# cfg = Namespace(**json.load(open(cfg_path)))
# os.makedirs(cfg.log_dir, exist_ok=True)
# os.makedirs(cfg.model_dir, exist_ok=True)

# model = PPOAgent(env=env, lr_actor=0.001, lr_critic=0.001, gamma=0.99, K_epochs=3,
#                  eps_clip=0.2, conf=cfg, continuous_action=continuous_action)
# model.train(episode_num=10000, update_interval=100, render_interval=5000, save_interval=3000)


cfg_path = os.path.join(path_filler('config'), 'rlf110_ppo_continuouscfg.json')
kwargs = json.load(open(cfg_path))
opt = Namespace(**kwargs)
print(opt)


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


def main():
    import torch
    import numpy as np

    envname = 'f110th'
    write = opt.write  # Use SummaryWriter to record the training.
    render = opt.render

    kwargs['state_dim'] = 54  # env.observation_space.shape[0]
    kwargs['action_dim'] = env.action_space.shape[0]
    kwargs['env_with_Dead'] = False
    max_action = float(env.action_space.high[0])
    max_steps = np.inf
    T_horizon = opt.T_horizon  # lenth of long trajectory

    Max_train_steps = opt.Max_train_steps
    save_interval = opt.save_interval  # in steps
    eval_interval = opt.eval_interval  # in steps

    random_seed = 63
    print("Random Seed: {}".format(random_seed))
    torch.manual_seed(random_seed)
    # env.seed(random_seed)
    # eval_env.seed(random_seed)
    np.random.seed(random_seed)

    if write:
        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
        writepath = 'runs/{}'.format(envname) + timenow
        if os.path.exists(writepath): shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)

    if not os.path.exists('model'): os.mkdir('model')
    model = PPO(**kwargs)
    # if opt.Loadmodel: model.load(opt.ModelIdex)

    traj_lenth = 0
    total_steps = 0
    restart_wp = None
    while total_steps < Max_train_steps:
        s, done, steps, ep_r = env.reset(traceback_restart=False), False, 0, 0

        while not done:
            traj_lenth += 1
            steps += 1

            if render:
                env.render()
                a, logprob_a = model.evaluate(s)
            else:
                a, logprob_a = model.select_action(s)

            act = policy(a, max_action)  # [0,1] to [-max,max]
            s_prime, r, done, info = env.step(act)
            # r = Reward_adapter(r, EnvIdex)

            if done and steps != max_steps:
                dw = True
                # still have exception: dead or win at _max_episode_steps will not be regard as dw.
                # Thus, decide dw according to reward signal of each game is better.  dw = done_adapter(r)
            else:
                dw = False

            model.put_data((s, a, r, s_prime, logprob_a, done, dw))
            s = s_prime
            ep_r += r

            if not render:
                if traj_lenth % T_horizon == 0:
                    model.train()
                    traj_lenth = 0

            if total_steps % eval_interval == 0:
                score = evaluate_policy(eval_env, model, True, max_steps, max_action)
                if write:
                    writer.add_scalar('ep_r_insteps', score, global_step=total_steps)
                print('EnvName:', envname, 'seed:', random_seed, 'steps: {}k'.format(int(total_steps / 1000)), 'score:',
                      score)
            total_steps += 1

            if total_steps % save_interval == 0:
                model.save(total_steps)

    env.close()


if __name__ == '__main__':
    main()

# eval
# model_name = 'ppo_speed4_action3/models/bestmodel_step78000'
# model.load(filename=model_name)
# model.eval(render=True)
