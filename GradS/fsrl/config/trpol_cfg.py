from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class TrainCfg:
    # general task params
    task: str = "SafetyCarGoal1Gymnasium-v0"

    safety_gymnasium_task: bool = True
    velocity_bound = 0.5
    velocity_lower_bound: float = 0.2
    concat_dim = 5 # useless
    # velo_thres = 50
    # task config
    cost_0_redundant_num: int = 1 # collision
    cost_1_redundant_num: int = 8
    cost_2_redundant_num: int = 0

    thre_0: float = 25
    thre_1: float = 25
    thre_2: float = 25

    cost_limit: Tuple[float, ...] = (50, 50, 50, 50, 20, 20)
    device: str = "cpu"
    thread: int = 4  # if use "cpu" to train
    seed: int = 50
    use_default_cfg: bool = False

    vanilla: bool = True
    random_con_selection: bool = False
    gradient_shaping: bool = False
    max_singleAC: bool = False

    # algorithm params
    conditioned_sigma: bool = True
    lr: float = 5e-4
    hidden_sizes: Tuple[int, ...] = (128, 128)
    unbounded: bool = False
    last_layer_scale: bool = False
    # PPO specific arguments
    target_kl: float = 0.001
    backtrack_coeff: float = 0.8
    max_backtracks: int = 10
    optim_critic_iters: int = 20
    gae_lambda: float = 0.95
    norm_adv: bool = True  # good for improving training stability
    # Lagrangian specific arguments
    use_lagrangian: bool = True
    lagrangian_pid: Tuple[float, ...] = (0.2, 0.002, 0.1) # (0.05, 0.0005, 0.1)
    rescaling: bool = True
    # Base policy common arguments
    gamma: float = 0.99
    max_batchsize: int = 99999
    rew_norm: bool = False
    deterministic_eval: bool = True
    action_scaling: bool = True
    action_bound_method: str = "clip"
    # collecting params
    epoch: int = 400
    episode_per_collect: int = 20
    step_per_epoch: int = 10000
    repeat_per_collect: int = 4  # increasing this can improve efficiency, but less stability
    buffer_size: int = 100000
    worker: str = "ShmemVectorEnv"
    training_num: int = 20
    testing_num: int = 10
    # general params
    # batch-size >> steps per collect means calculating all data in one singe forward.
    batch_size: int = 99999
    reward_threshold: float = 10000  # for early stop purpose
    save_interval: int = 4
    resume: bool = False  # TODO
    save_ckpt: bool = True  # set this to True to save the policy model
    verbose: bool = True
    render: bool = False
    # logger params
    logdir: str = "logs"
    project: str = "11-11-CG1-Redundant-trpo"
    group: Optional[str] = None
    name: Optional[str] = None
    prefix: Optional[str] = "trpol"
    suffix: Optional[str] = ""


# bullet-safety-gym task default configs


@dataclass
class Bullet1MCfg(TrainCfg):
    epoch: int = 100


@dataclass
class Bullet5MCfg(TrainCfg):
    epoch: int = 500


@dataclass
class Bullet10MCfg(TrainCfg):
    epoch: int = 1000


# safety gymnasium task default configs


@dataclass
class MujocoBaseCfg(TrainCfg):
    task: str = "SafetyPointCircle1-v0"
    epoch: int = 250
    cost_limit: float = 25
    # collecting params
    episode_per_collect: int = 20
    step_per_epoch: int = 20000
    repeat_per_collect: int = 4  # increasing this can improve efficiency, but less stability


@dataclass
class Mujoco2MCfg(MujocoBaseCfg):
    epoch: int = 100


@dataclass
class Mujoco20MCfg(MujocoBaseCfg):
    epoch: int = 1000


@dataclass
class Mujoco10MCfg(MujocoBaseCfg):
    epoch: int = 500