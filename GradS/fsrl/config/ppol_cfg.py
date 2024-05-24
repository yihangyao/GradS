from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class TrainCfg:
    # general task params
    task: str = "SafetyBallCircle-v0" #
    short_task: str = "DC-v3"

    safety_gymnasium_task: bool = False
    velocity_bound = 0.7
    velocity_lower_bound: float = 0.2
    concat_dim = 5 # useless
    # velo_thres = 50
    # task config
    cost_0_redundant_num: int = 1 
    cost_1_redundant_num: int = 1
    cost_2_redundant_num: int = 1

    thre_0: float = 50
    thre_1: float = 50
    thre_2: float = 50

    cost_limit: Tuple[float, ...] = 25 # , 20, 100, 20, 100, 20, 100
    device: str = "cpu"
    thread: int = 4  # if use "cpu" to train
    seed: int = 50
    use_default_cfg: bool = True

    vanilla: bool = False
    random_con_selection: bool = False
    gradient_shaping: bool = True
    max_singleAC: bool = False

    # algorithm params
    conditioned_sigma: bool = True
    lr: float = 1e-4
    hidden_sizes: Tuple[int, ...] = (256, 256)
    unbounded: bool = False
    last_layer_scale: bool = False
    # PPO specific arguments
    target_kl: float = 0.02
    vf_coef: float = 0.25
    max_grad_norm: Optional[float] = 0.5
    gae_lambda: float = 0.95
    eps_clip: float = 0.2
    dual_clip: Optional[float] = None
    value_clip: bool = False  # no need
    norm_adv: bool = True  # good for improving training stability
    recompute_adv: bool = False
    # Lagrangian specific arguments
    use_lagrangian: bool = True
    lagrangian_pid: Tuple[float, ...] = (0.15, 0.005, 0.1) # (0.05, 0.0005, 0.1) (0.2, 0.001, 0.1)
    rescaling: bool = True
    # Base policy common arguments
    gamma: float = 0.99
    max_batchsize: int = 100000
    rew_norm: bool = False  # no need, it will slow down training and decrease final perf
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
    batch_size: int = 256
    reward_threshold: float = 10000  # for early stop purpose
    save_interval: int = 4
    resume: bool = False  # TODO
    save_ckpt: bool = True  # set this to True to save the policy model
    verbose: bool = True
    render: bool = False
    # logger params
    logdir: str = "logs"
    project: str = "BC-v3-GradS(PPO)" # 
    group: Optional[str] = None
    name: Optional[str] = None
    prefix: Optional[str] = "ppol" # 
    suffix: Optional[str] = "-0524-1"

@dataclass
class BC_v2(TrainCfg):
    task: str = "SafetyBallCircle-v0" #
    epoch: int = 400
    short_task: str = "BC-v2"
    thre_0: float = 50
    thre_1: float = 50
    thre_2: float = 25
    cost_0_redundant_num: int = 1 
    cost_1_redundant_num: int = 1
    cost_2_redundant_num: int = 0

@dataclass
class CC_v2(TrainCfg):
    task: str = "SafetyCarCircle-v0" #
    epoch: int = 400
    short_task: str = "CC-v2"
    thre_0: float = 50
    thre_1: float = 50
    thre_2: float = 20
    cost_0_redundant_num: int = 1 
    cost_1_redundant_num: int = 1
    cost_2_redundant_num: int = 0

@dataclass
class DC_v2(TrainCfg):
    task: str = "SafetyDroneCircle-v0" #
    epoch: int = 600
    short_task: str = "DC-v2"
    thre_0: float = 250
    thre_1: float = 50
    thre_2: float = 50
    cost_0_redundant_num: int = 1 
    cost_1_redundant_num: int = 1
    cost_2_redundant_num: int = 0

@dataclass
class BC_v3(TrainCfg):
    task: str = "SafetyBallCircle-v0" #
    epoch: int = 400
    short_task: str = "BC-v3"
    thre_0: float = 50
    thre_1: float = 50
    thre_2: float = 25
    cost_0_redundant_num: int = 1 # collision
    cost_1_redundant_num: int = 1
    cost_2_redundant_num: int = 1

@dataclass
class CC_v3(TrainCfg):
    task: str = "SafetyCarCircle-v0" #
    epoch: int = 400
    short_task: str = "CC-v3"
    thre_0: float = 50
    thre_1: float = 50
    thre_2: float = 20
    cost_0_redundant_num: int = 1 # collision
    cost_1_redundant_num: int = 1
    cost_2_redundant_num: int = 1

@dataclass
class DC_v3(TrainCfg):
    task: str = "SafetyDroneCircle-v0" #
    epoch: int = 600
    short_task: str = "DC-v3"
    thre_0: float = 250
    thre_1: float = 50
    thre_2: float = 50
    cost_0_redundant_num: int = 1 # collision
    cost_1_redundant_num: int = 1
    cost_2_redundant_num: int = 1


# safety gymnasium task default configs
@dataclass
class MujocoBaseCfg(TrainCfg):
    # task: str = "SafetyPointCircle1-v0"
    epoch: int = 250
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
