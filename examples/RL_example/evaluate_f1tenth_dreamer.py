import os.path

import fire
import numpy as np

from common.policys import NetworkPolicy
from pydreamer.models import Dreamer
import argparse
from pydreamer import tools
from distutils.util import strtobool
from f110_rlenv import create_dictObs_eval_env
from pydreamer.preprocessing import Preprocessor, WorkerInfoPreprocess
from utils import path_filler
import torch
from f110_rlenv import Lidar_Manager


def get_dreamer_cfg():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--configs', nargs='+', required=True)
    args, remaining = parser.parse_known_args()

    # Config from YAML
    conf = {}
    configs = tools.read_yamls('./config')
    # for name in args.configs:
    name = 'defaults,f110env'
    if ',' in name:
        for n in name.split(','):
            conf.update(configs[n])
    else:
        conf.update(configs[name])
    conf['run_name'] = 'f110env'
    # Override config from command-line
    # parser = argparse.ArgumentParser()
    for key, value in conf.items():
        type_ = type(value) if value is not None else str
        if type_ == bool:
            type_ = lambda x: bool(strtobool(x))
        parser.add_argument(f'--{key}', type=type_, default=value)
    # parser.add_argument(f'--run_name', default='cartPole1')
    conf = parser.parse_args(remaining)
    return conf


def evaluate_dreamer_model(checkpoint_path=None):
    conf = get_dreamer_cfg()
    preprocess = Preprocessor(image_categorical=conf.image_channels if conf.image_categorical else None,
                              image_key=conf.image_key,
                              map_categorical=conf.map_channels if conf.map_categorical else None,
                              map_key=conf.map_key,
                              action_dim=conf.action_dim,
                              clip_rewards=conf.clip_rewards,
                              amp=False)
    dreamer_model = Dreamer(conf)
    if not checkpoint_path:
        mlrun_dir = path_filler('mlruns/0')
        for run_id in os.listdir(mlrun_dir):
            print(run_id)
            if os.path.isdir(os.path.join(mlrun_dir, run_id)):
                path = os.path.join(mlrun_dir, run_id, 'artifacts/checkpoints/latest.pt')
                checkpoint_path = path
                print(f'load checkpoint from {run_id}')
                break
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    dreamer_model.load_state_dict(checkpoint['model_state_dict'])
    dreamer_policy = NetworkPolicy(dreamer_model, preprocess)
    env = create_dictObs_eval_env()

    for i in range(5):
        obs = env.reset()
        done = False
        # env.render()
        i = 0
        while not done:
            i += 1
            env.render()

            ##### use policy ######
            action, metric = dreamer_policy(obs)
            obs, step_reward, done, info = env.step(action)
            if i % 10 == 0:
                # print(f'step_reward: {step_reward}')
                # print(min(obs['vecobs']))
                # print(env.wpManager.get_lateral_error(obs['raw_obs']))
                # print(step_reward)
                pass
        print('finish one episode')


def draw_two_lidar(vecobs, pred_vecobs):
    import matplotlib.pyplot as plt
    import cv2
    lidar_manager = Lidar_Manager(scan_dim=108, window_H=250, xy_range=15)
    pred_vecobs_map = lidar_manager.update_scan2map(pred_vecobs, color='b')
    # pred_vecobs_map *= 0.5
    vecobs_map = lidar_manager.update_scan2map(vecobs)
    # vecobs_map *= -0.8
    maps = np.hstack([pred_vecobs_map, vecobs_map])

    ax = plt.subplot(1, 1, 1)
    ax.plot(vecobs, '-r', label='lidar scans')
    ax.plot(pred_vecobs, '-b', label='lidar scans from world model(imagination)')
    plt.legend()
    plt.show()

    cv2.imshow('lidar reconstruction', maps)
    cv2.waitKey(5)


def get_reconstruct_lidar():
    import matplotlib.pyplot as plt
    import cv2
    lidar_manager = Lidar_Manager(scan_dim=108, window_H=250, xy_range=15)

    mlrun_dir = path_filler('mlruns/0')
    for run_id in os.listdir(mlrun_dir):
        print(run_id)
        if os.path.isdir(os.path.join(mlrun_dir, run_id)):
            npz_dir = os.path.join(mlrun_dir, run_id, 'artifacts/d2_wm_closed_eval')
            latest_npz = '0003000.npz'
            print(f'load npz from {os.path.join(npz_dir, latest_npz)}')
            # with open(os.path.join(npz_dir, latest_npz)) as f:
            fdata = np.load(os.path.join(npz_dir, latest_npz), mmap_mode='r')
            # import ipdb; ipdb.set_trace()
    pred_loss = fdata['loss_vecobs'][0]  # original(1, 1100)
    vecobs = fdata['vecobs'][0]
    pred_vecobs = fdata['vecobs_pred'][0]
    # sorted_loss_idx = np.argsort(pred_loss)
    sorted_loss_idx = range(len(pred_loss))
    init = True
    for i in sorted_loss_idx[40:-600:20]:
        if init:
            pred_vecobs_map = lidar_manager.update_scan2map(pred_vecobs[i], color='b')
            vecobs_map = lidar_manager.update_scan2map(vecobs[i])
            init = False
        else:
            pred_vecobs_map = np.hstack([pred_vecobs_map, lidar_manager.update_scan2map(pred_vecobs[i], color='b')])
            vecobs_map = np.hstack([vecobs_map, lidar_manager.update_scan2map(vecobs[i])])
    maps = np.vstack([pred_vecobs_map, vecobs_map])
    cv2.imshow('lidar reconstruction', maps)
    cv2.waitKey(10000)

if __name__ == '__main__':
    fire.Fire({
        'eval_visual_env': evaluate_dreamer_model,
        'see_dream': get_reconstruct_lidar
    })
