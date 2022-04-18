# -- coding: utf-8 --
import json
import os
import yaml
from argparse import Namespace
from utils import render_callback, path_filler
from f110_rlenv import F110Env_Discrete_Action, F110Env_Continuous_Action
from baselineAgents.ppo_agent import PPOAgent

with open('./config_example_map.yaml') as file:
    conf_dict = yaml.load(file, Loader=yaml.FullLoader)
conf = Namespace(**conf_dict)

continuous_action = True
if continuous_action:
    env = F110Env_Continuous_Action(conf=conf)
else:
    env = F110Env_Discrete_Action(conf=conf)
print('continuous_action', continuous_action)
env.f110.add_render_callback(render_callback)

# train
task = 'ppo'
cfg_path = os.path.join(path_filler('config'), f'rlf110_{task}cfg.json')
cfg = Namespace(**json.load(open(cfg_path)))
os.makedirs(cfg.log_dir, exist_ok=True)
os.makedirs(cfg.model_dir, exist_ok=True)

model = PPOAgent(env=env, lr_actor=0.001, lr_critic=0.001, gamma=0.99, K_epochs=3,
                 eps_clip=0.2, conf=cfg, continuous_action=continuous_action)
model.train(episode_num=10000, update_interval=100, render_interval=5000, save_interval=3000)


# eval
model_name = 'ppo_speed4_action3/models/bestmodel_step78000'
model.load(filename=model_name)
model.eval(render=True)
