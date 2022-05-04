# -- coding: utf-8 --
import json
import os
import numpy as np
import yaml
import fire

work_dir = os.path.abspath('..')
cfg_dir = os.path.abspath('.')
map_dir = os.path.join(cfg_dir, 'maps')

### change sim_cfg_file(yaml) ###
def get_sim_cfg(map_name='example_map'):
    # if not map_name:
    #     raise TypeError('Must give map name')
    with open('./maps/config_example_map.yaml') as f:
        sim_cfg_temp = yaml.safe_load(f)
    
    sim_cfg_temp['map_path'] = os.path.join(map_dir, map_name)
    sim_cfg_temp['wpt_path'] = os.path.join(map_dir, map_name+'_wp.csv')

    with open(os.path.join(map_dir, 'config_' + map_name + '.yaml'), 'w') as f:
        yaml.dump(sim_cfg_temp, f)


def path_filler(path=None):
    abs_path = os.path.abspath(os.path.join('.', path))
    return abs_path


def fill_cfg(task):
    upper_dir = os.path.abspath('..')
    result_dir = path_filler('result')
    default_cfg = dict(
        task_name=task,
        result_dir=os.path.join(result_dir, task),
        log_dir=os.path.join(result_dir, task, 'logs'),
        model_dir=os.path.join(upper_dir, 'evaluate_model')
    )
    return default_cfg


def get_rlf110_cfg(task='ddqn', cfg=None):
    if not cfg:
        cfg = fill_cfg(task)
    os.makedirs(os.path.join(path_filler('config')), exist_ok=True)
    json.dump(cfg, open(os.path.join(path_filler('config'), f'rlf110_{task}cfg.json'), 'w'), indent=4)

    """
        no_terminal (bool, optional): _description_. Defaults to False.
        env_time_limit (int, optional): _description_. Defaults to 100000.
        env_action_repeat (int, optional): _description_. Defaults to 1.
        sim_cfg_file (str, optional): _description_. Defaults to './config_example_map.yaml'.
        render (bool, optional): _description_. Defaults to False.
        continuous_action (bool, optional): _description_. Defaults to False.
        display_lidar (bool, optional): _description_. Defaults to False.
        limited_time = False
    """

def get_rlf110_env_cfg(map_name='example_map'):
    env_cfg = dict(
        speed=3,
        obs_shape=54,
        continuous_action=True,
        sim_cfg_file = os.path.join(map_dir, 'config_' + map_name + '.yaml'),
        limited_time=False,
        no_terminal=False,
        env_time_limit=0,
        env_action_repeat=1,
        render_env=True,
        display_lidar=False,
        dictObs=True
    )
    print('get_rl_env_cfg')
    json.dump(env_cfg, open(os.path.join(cfg_dir, 'rlf110_env_cfg.json'), 'w'), indent=4)

if __name__ == '__main__':
    # get_rlf110_cfg(task='ddqn')
    fire.Fire({
        'get_sim_cfg': get_sim_cfg,
        'get_env_cfg': get_rlf110_env_cfg,
        'get_ppo_discrete_cfg': get_rlf110_cfg(task='ppo')
    })
    
