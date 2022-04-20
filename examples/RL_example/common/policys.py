from typing import Dict, List, Optional, Tuple
import torch
import numpy as np
from baselineAgents.gap_follow_agent import Gap_follower
from pydreamer.models import Dreamer
from pydreamer.preprocessing import Preprocessor
from pydreamer.models import *
from pydreamer.models.functions import map_structure
from pydreamer.preprocessing import Preprocessor
from pydreamer.tools import *


class RandomPolicy:
    def __init__(self, action_space):
        self.action_space = action_space

    def __call__(self, obs) -> Tuple[int, dict]:
        return self.action_space.sample(), {}


class PurePursuitPolicy:
    def __init__(self):
        pass

    def __call__(self, obs):
        pass

class GapFollowPolicy:
    def __init__(self):
        self.model = Gap_follower()

    def __call__(self, raw_obs):
        scan = raw_obs['scans'][0]
        action, target = self.model.planning(scan)
        return action, {'target_idx': target}
        


class NetworkPolicy:
    def __init__(self, model: Dreamer, preprocess: Preprocessor):
        self.model = model
        self.preprocess = preprocess
        self.state = model.init_state(1)

    def __call__(self, obs) -> Tuple[np.ndarray, dict]:
        batch = self.preprocess.apply(obs, expandTB=True)
        obs_model: Dict[str, Tensor] = map_structure(batch, torch.from_numpy)  # type: ignore

        with torch.no_grad():
            action_distr, new_state, metrics = self.model.forward(obs_model, self.state)
            action = action_distr.sample()
            self.state = new_state

        metrics = {k: v.item() for k, v in metrics.items()}
        metrics.update(action_prob=action_distr.log_prob(action).exp().mean().item(),
                       policy_entropy=action_distr.entropy().mean().item())

        action = action.squeeze()  # (1,1,A) => A
        # add as type to adjust to the f110RL env
        return action.numpy().astype(np.int64), metrics

