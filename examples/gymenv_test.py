import gym
from common.policys import RandomPolicy
from typing import NamedTuple
from argparse import Namespace
import yaml

# with open('config_example_map.yaml') as file:
#     conf_dict = yaml.load(file, Loader=yaml.FullLoader)
# conf = Namespace(**conf_dict)
#
# env = gym.make('f110_gym:f110-v0', **conf_dict)
# p = RandomPolicy(env.action_space)
#
# s = env.reset()
# done = False
#
# while not done:
#     # env.render()
#     a, _ = p(s)
#     s, reward, done, info = env.step(a)
#
# env.close()

env = gym.make("BipedalWalker-v3")
print(env.action_space.shape)