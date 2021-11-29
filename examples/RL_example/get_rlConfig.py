# -- coding: utf-8 --
import json
import os
import numpy as np


def path_filler(path):
    abs_path = os.path.abspath(os.path.join('.', path))
    return abs_path

def fill_cfg(task):
    result_dir = path_filler('result')
    default_cfg = dict(
        task_name=task,
        result_dir=os.path.join(result_dir, task),
        log_dir=os.path.join(result_dir, task, 'logs'),
        model_dir=os.path.join(result_dir, task, 'models')
    )
    return default_cfg


def get_rlf110_cfg(task='ddqn'):
    cfg = fill_cfg(task)
    os.makedirs(os.path.join(path_filler('config')) ,exist_ok=True)
    json.dump(cfg, open(os.path.join(path_filler('config'), f'rlf110_{task}cfg.json'), 'w'), indent=4)


if __name__ == '__main__':
    get_rlf110_cfg(task='ppo')
