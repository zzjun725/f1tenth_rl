# -- coding: utf-8 --
import json
import os
import yaml
from argparse import Namespace
from utils import render_callback, path_filler
from f110_rlenv import F110Env_Discrete_Action
from d3qn_agent import D3QNAgent

with open('./config_example_map.yaml') as file:
    conf_dict = yaml.load(file, Loader=yaml.FullLoader)
conf = Namespace(**conf_dict)
env = F110Env_Discrete_Action(conf=conf)
env.f110.add_render_callback(render_callback)

# train
task = 'ddqn'
cfg_path = os.path.join(path_filler('config'), f'RlF110_{task}cfg.json')
cfg = Namespace(**json.load(open(cfg_path)))
os.makedirs(cfg.log_dir, exist_ok=True)
os.makedirs(cfg.model_dir, exist_ok=True)

# parameters
steps = 200000
memory_size = 1000
batch_size = 64
target_update = 100
epsilon_decay = 1 / 20000

agent = D3QNAgent(env, memory_size, batch_size, target_update, epsilon_decay, conf=cfg)
# agent.train(steps, render_interval=1000)

# eval
# model_name = 'step60000.pkl'
# agent.eval(reload_model=True, render=True, model=model_name)