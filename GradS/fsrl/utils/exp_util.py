# flake8: noqa

import os
import os.path as osp
import random
import uuid
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import yaml

from fsrl.utils.logger.logger_util import colorize
import gymnasium as gym


def seed_all(seed=1029, others: Optional[list] = None) -> None:
    """Fix the seeds of `random`, `numpy`, `torch` and the input `others` object.

    :param int seed: defaults to 1029
    :param Optional[list] others: other objects that want to be seeded, defaults to None
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    # torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if others is not None:
        if hasattr(others, "seed"):
            others.seed(seed)
            return True
        try:
            for item in others:
                if hasattr(item, "seed"):
                    item.seed(seed)
        except:
            pass


def get_cfg_value(config, key):
    if key in config:
        value = config[key]
        if isinstance(value, list):
            suffix = ""
            for i in value:
                suffix += str(i)
            return suffix
        return str(value)
    for k in config.keys():
        if isinstance(config[k], dict):
            res = get_cfg_value(config[k], key)
            if res is not None:
                return res
    return "None"


def load_config_and_model(path: str, best: bool = False):
    """
    Load the configuration and trained model from a specified directory.

    :param path: the directory path where the configuration and trained model are stored.
    :param best: whether to load the best-performing model or the most recent one.
        Defaults to False.

    :return: a tuple containing the configuration dictionary and the trained model.
    :raises ValueError: if the specified directory does not exist.
    """
    if osp.exists(path):
        config_file = osp.join(path, "config.yaml")
        print(f"load config from {config_file}")
        with open(config_file) as f:
            config = yaml.load(f.read(), Loader=yaml.FullLoader)
        model_file = "model.pt"
        if best:
            model_file = "model_best.pt"
        model_path = osp.join(path, "checkpoint/" + model_file)
        print(f"load model from {model_path}")
        model = torch.load(model_path)
        return config, model
    else:
        raise ValueError(f"{path} doesn't exist!")


# naming utils


def to_string(values):
    """
    Recursively convert a sequence or dictionary of values to a string representation.

    :param values: the sequence or dictionary of values to be converted to a string.
    :return: a string representation of the input values.
    """
    name = ""
    if isinstance(values, Sequence) and not isinstance(values, str):
        for i, v in enumerate(values):
            prefix = "" if i == 0 else "_"
            name += prefix + to_string(v)
        return name
    elif isinstance(values, Dict):
        for i, k in enumerate(sorted(values.keys())):
            prefix = "" if i == 0 else "_"
            name += prefix + to_string(values[k])
        return name
    else:
        return str(values)


DEFAULT_SKIP_KEY = [
    "task", "reward_threshold", "logdir", "worker", "project", "group", "name", "prefix",
    "suffix", "save_interval", "render", "verbose", "save_ckpt", "training_num",
    "testing_num", "epoch", "device", "thread"
]

DEFAULT_KEY_ABBRE = {
    "cost_limit": "cost",
    "mstep_iter_num": "mnum",
    "estep_iter_num": "enum",
    "estep_kl": "ekl",
    "mstep_kl_mu": "kl_mu",
    "mstep_kl_std": "kl_std",
    "mstep_dual_lr": "mlr",
    "estep_dual_lr": "elr",
    "update_per_step": "update"
}


def auto_name(
    default_cfg: dict,
    current_cfg: dict,
    prefix: str = "",
    suffix: str = "",
    skip_keys: list = DEFAULT_SKIP_KEY,
    key_abbre: dict = DEFAULT_KEY_ABBRE
) -> str:
    """Automatic generate the name by comparing the current config with the default one.

    :param dict default_cfg: a dictionary containing the default configuration values.
    :param dict current_cfg: a dictionary containing the current configuration values.
    :param str prefix: (optional) a string to be added at the beginning of the generated
        name.
    :param str suffix: (optional) a string to be added at the end of the generated name.
    :param list skip_keys: (optional) a list of keys to be skipped when generating the
        name.
    :param dict key_abbre: (optional) a dictionary containing abbreviations for keys in
        the generated name.

    :return str: a string representing the generated experiment name.
    """
    name = prefix
    for i, k in enumerate(sorted(default_cfg.keys())):
        if default_cfg[k] == current_cfg[k] or k in skip_keys:
            continue
        prefix = "_" if len(name) else ""
        value = to_string(current_cfg[k])
        # replace the name with abbreviation if key has abbreviation in key_abbre
        if k in key_abbre:
            k = key_abbre[k]
        # Add the key-value pair to the name variable with the prefix
        name += prefix + k + value
    if len(suffix):
        name = name + "_" + suffix if len(name) else suffix

    name = "default" if not len(name) else name
    name = f"{name}-{str(uuid.uuid4())[:4]}"
    return name

class redundant_wrapper(gym.Wrapper):
    def __init__(self, env,
                 cost_0_redunt_num = 8, # info['cost_velocity_violation']
                 cost_1_redunt_num = 2, # info['cost_outside_bounds']
                 cost_2_redunt_num = 0, # info['cost_velocity_violation_low']
                 ):
        gym.Wrapper.__init__(self, env)
        self.cost_0_redunt_num = cost_0_redunt_num
        self.cost_1_redunt_num = cost_1_redunt_num
        self.cost_2_redunt_num = cost_2_redunt_num

    def reset(self):
        obs, info = self.env.reset()
        return obs, info

    def step(self, action):

        obs_next, reward, terminated, truncated, info = self.env.step(action)
        cost_0 = info['cost_velocity_violation']
        cost_1 = info['cost_outside_bounds']
        cost_2 = info['cost_velocity_violation_low']
        m_cost = []

        for i in range(self.cost_0_redunt_num):
            info['sub_cost_'+str(i)] = cost_0
            m_cost.append(cost_0)
        for j in range(self.cost_1_redunt_num):
            info['sub_cost_'+str(j+self.cost_0_redunt_num)] = cost_1
            m_cost.append(cost_1)
        for k in range(self.cost_2_redunt_num):
            info['sub_cost_'+str(k+self.cost_0_redunt_num+self.cost_1_redunt_num)] = cost_2
            m_cost.append(cost_2)

        info["m_cost"] = m_cost

        return obs_next, reward, terminated, truncated, info
    

class vel_wrapper(gym.Wrapper):
    def __init__(self, env,
                 velocity_bound,
                 velocity_lower_bound,
                 collision_redundant_num,
                 velocity_redundant_num,
                 velocity_lower_redundant_num,
                 concat_dim = 5):
        gym.Wrapper.__init__(self, env)

        # for safetygymnasium only
        self.velocity_bound = velocity_bound
        self.velocity_lower_bound = velocity_lower_bound
        self.velocity_redundant_num = velocity_redundant_num
        self.collision_redundant_num = collision_redundant_num
        self.velocity_lower_redundant_num = velocity_lower_redundant_num
        self.concat_dim = concat_dim
        self.env.observation_space.shape

    def reset(self):
        obs, info = self.env.reset()
        # phi_0 = self.phi_0.forward(obs)
        # info['cost_env'] = 0.
        # info['cost'] = 0.
        # info['cost_shaped'] = 0.
        agent_vel = self.env.task.agent.vel
        vel_norm = np.linalg.norm(agent_vel[:2])

        if vel_norm > self.velocity_bound:
            vel_cost = 1
        else:
            vel_cost = 0

        # sub_cost_index = int(info['num_sub_cost'])
        # for i in range(self.velocity_redundant_num):
        #     info["sub_cost_"+str(i+sub_cost_index)] = vel_cost # vel_cost

        # obs = np.concatenate((obs, state_vel_ndarray))

        return obs, info

    def step(self, action):

        obs_next, reward, terminated, truncated, info = self.env.step(action)
        agent_vel = self.env.task.agent.vel
        vel_norm = np.linalg.norm(agent_vel[:2])
        if vel_norm > self.velocity_bound:
            vel_cost = 0.1
        else:
            vel_cost = 0

        if vel_norm < self.velocity_lower_bound:
            vel_lower_cost = 0.1
        else:
            vel_lower_cost = 0

        # sub_cost_index = int(info['num_sub_cost'])
        cost_index = 0
        cost_collision = info["sub_cost_0"]
        # original (L4DC version): 0: collision, 1: high speed, 2: low speed

        for _ in range(self.velocity_redundant_num):
            info["sub_cost_"+str(cost_index)] = vel_cost # vel_cost
            cost_index += 1

        for _ in range(self.collision_redundant_num):
            info["sub_cost_"+str(cost_index)] = cost_collision
            cost_index += 1 

        for _ in range(self.velocity_lower_redundant_num):
            info["sub_cost_"+str(cost_index)] = vel_lower_cost # vel_cost
            cost_index += 1

        info['cost'] = vel_cost

        return obs_next, reward, terminated, truncated, info
