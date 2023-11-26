from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class TrainCfg:
    # general task params
    task: str = "SafetyBallCircle-v0"
    safety_gymnasium_task: bool = False
    velocity_bound = 1.5
    velo_thres = 1000
    # task config
    cost_0_redundant_num: int = 1 # velocity
    cost_1_redundant_num: int = 1
    cost_2_redundant_num: int = 1

    thre_0: float = 50.0
    thre_1: float = 20.0
    thre_2: float = 50.0

    cost_limit: Tuple[float, ...] = (200.0, 200, 200, 200, 20, 20.0)
    device: str = "cpu"
    thread: int = 4  # if use "cpu" to train
    seed: int = 50
    use_default_cfg: bool = True

    vanilla: bool = False
    random_con_selection: bool = False
    gradient_shaping: bool = False
    max_singleAC: bool = True

    # algorithm params
    actor_lr: float = 5e-4
    critic_lr: float = 1e-3
    hidden_sizes: Tuple[int, ...] = (128, 128)
    tau: float = 0.05
    exploration_noise: float = 0.1
    n_step: int = 5
    # Lagrangian specific arguments
    use_lagrangian: bool = True
    lagrangian_pid: Tuple[float, ...] = (0.1, 0.002, 0.1) # (0.05, 0.0005, 0.1)
    rescaling: bool = True
    # Base policy common arguments
    gamma: float = 0.97
    deterministic_eval: bool = True
    action_scaling: bool = True
    action_bound_method: str = "clip"
    # collecting params
    epoch: int = 200
    episode_per_collect: int = 2
    step_per_epoch: int = 10000
    update_per_step: float = 0.2
    buffer_size: int = 100000
    worker: str = "ShmemVectorEnv"
    training_num: int = 10
    testing_num: int = 2
    # general params
    batch_size: int = 256
    reward_threshold: float = 10000  # for early stop purpose
    save_interval: int = 4
    resume: bool = False  # TODO
    save_ckpt: bool = True  # set this to True to save the policy model
    verbose: bool = True
    render: bool = False
    # logger params
    logdir: str = "logs"
    project: str = "10-21-Conflicting-BC-ddpgl"
    group: Optional[str] = None
    name: Optional[str] = None
    prefix: Optional[str] = "ddpgl"
    suffix: Optional[str] = ""
    

# bullet-safety-gym task default configs


@dataclass
class Bullet1MCfg(TrainCfg):
    epoch: int = 300


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
    gamma: float = 0.99
    n_step: int = 3
    # collecting params
    step_per_epoch: int = 20000
    episode_per_collect = 5
    buffer_size: int = 800000


@dataclass
class Mujoco2MCfg(MujocoBaseCfg):
    epoch: int = 100


@dataclass
class Mujoco20MCfg(MujocoBaseCfg):
    epoch: int = 1000


@dataclass
class Mujoco10MCfg(MujocoBaseCfg):
    epoch: int = 500
