import os
from dataclasses import asdict

import bullet_safety_gym

try:
    import safety_gymnasium
except ImportError:
    print("safety_gymnasium is not found.")
import gymnasium as gym
import pyrallis
from tianshou.env import BaseVectorEnv, ShmemVectorEnv, SubprocVectorEnv

from fsrl.agent import SACLagAgent
from fsrl.config.sacl_cfg import (
    Bullet1MCfg,
    Bullet5MCfg,
    Bullet10MCfg,
    Mujoco2MCfg,
    Mujoco10MCfg,
    Mujoco20MCfg,
    MujocoBaseCfg,
    TrainCfg,
)
from fsrl.utils import BaseLogger, TensorboardLogger, WandbLogger
from fsrl.utils.exp_util import auto_name, redundant_wrapper, vel_wrapper

TASK_TO_CFG = {
    # bullet safety gym tasks
    "SafetyCarRun-v0": Bullet1MCfg,
    "SafetyBallRun-v0": Bullet1MCfg,
    "SafetyBallCircle-v0": Bullet1MCfg,
    "SafetyCarCircle-v0": TrainCfg,
    "SafetyDroneRun-v0": TrainCfg,
    "SafetyAntRun-v0": TrainCfg,
    "SafetyDroneCircle-v0": Bullet5MCfg,
    "SafetyAntCircle-v0": Bullet10MCfg,
    # safety gymnasium tasks
    "SafetyPointCircle1Gymnasium-v0": Mujoco2MCfg,
    "SafetyPointCircle2Gymnasium-v0": Mujoco2MCfg,
    "SafetyCarCircle1Gymnasium-v0": Mujoco2MCfg,
    "SafetyCarCircle2Gymnasium-v0": Mujoco2MCfg,
    "SafetyPointGoal1Gymnasium-v0": MujocoBaseCfg,
    "SafetyPointGoal2Gymnasium-v0": MujocoBaseCfg,
    "SafetyPointButton1Gymnasium-v0": MujocoBaseCfg,
    "SafetyPointButton2Gymnasium-v0": MujocoBaseCfg,
    "SafetyPointPush1Gymnasium-v0": MujocoBaseCfg,
    "SafetyPointPush2Gymnasium-v0": MujocoBaseCfg,
    "SafetyCarGoal1Gymnasium-v0": MujocoBaseCfg,
    "SafetyCarGoal2Gymnasium-v0": MujocoBaseCfg,
    "SafetyCarButton1Gymnasium-v0": MujocoBaseCfg,
    "SafetyCarButton2Gymnasium-v0": MujocoBaseCfg,
    "SafetyCarPush1Gymnasium-v0": MujocoBaseCfg,
    "SafetyCarPush2Gymnasium-v0": MujocoBaseCfg,
    "SafetyHalfCheetahVelocityGymnasium-v1": MujocoBaseCfg,
    "SafetyHopperVelocityGymnasium-v1": MujocoBaseCfg,
    "SafetySwimmerVelocityGymnasium-v1": MujocoBaseCfg,
    "SafetyWalker2dVelocityGymnasium-v1": Mujoco10MCfg,
    "SafetyAntVelocityGymnasium-v1": Mujoco10MCfg,
    "SafetyHumanoidVelocityGymnasium-v1": Mujoco20MCfg,
}


@pyrallis.wrap()
def train(args: TrainCfg):

    task = args.task
    default_cfg = TASK_TO_CFG[task]() if task in TASK_TO_CFG else TrainCfg()
    # use the default configs instead of the input args.
    if args.use_default_cfg:
        default_cfg.task = args.task
        default_cfg.seed = args.seed
        default_cfg.device = args.device
        default_cfg.logdir = args.logdir
        default_cfg.project = args.project
        default_cfg.group = args.group
        default_cfg.suffix = args.suffix
        args = default_cfg

    cost_0_thres = args.thre_0
    cost_1_thres = args.thre_1
    cost_2_thres = args.thre_2

    cost_limit_new = []

    if args.safety_gymnasium_task:
        # cost_1_thres = args.velo_thres
        for _ in range(args.cost_0_redundant_num):
            cost_limit_new.append(cost_0_thres)
        for _ in range(args.cost_1_redundant_num):
            cost_limit_new.append(cost_1_thres)
        for _ in range(args.cost_2_redundant_num):
            cost_limit_new.append(cost_2_thres)

    else:
        for _ in range(args.cost_0_redundant_num):
            cost_limit_new.append(cost_0_thres)
        for _ in range(args.cost_1_redundant_num):
            cost_limit_new.append(cost_1_thres)
        for _ in range(args.cost_2_redundant_num):
            cost_limit_new.append(cost_2_thres)

    args.cost_limit = cost_limit_new

    # setup logger
    cfg = asdict(args)
    default_cfg = asdict(default_cfg)
    if args.name is None:
        args.name = auto_name(default_cfg, cfg, args.prefix, args.suffix)
    if args.group is None:
        args.group = args.task + "-cost-0-" + str(cost_0_thres) + "-cost-1-" + str(cost_1_thres) + "-cost-2-" + str(cost_2_thres) + \
              "-c0-" + str(args.cost_0_redundant_num) + \
              "-c1-" + str(args.cost_1_redundant_num) + "-c2-" + str(args.cost_2_redundant_num) + \
              "-V-" + str(args.vanilla) + "-R-" + str(args.random_con_selection) + \
              "-G-" + str(args.gradient_shaping) + "-M-" + str(args.max_singleAC)

    if args.safety_gymnasium_task:
        args.group  = "vel-bound-" + str(args.velocity_bound) + '-' +  args.group 

    if args.logdir is not None:
        args.logdir = os.path.join(args.logdir, args.project, args.group)
    logger = WandbLogger(cfg, args.project, args.group, args.name, args.logdir) # BaseLogger() #
    # logger = TensorboardLogger(args.logdir, log_txt=True, name=args.name)
    logger.save_config(cfg, verbose=args.verbose)

    if args.safety_gymnasium_task:
        demo_env = vel_wrapper(gym.make(args.task), 
                               velocity_bound=args.velocity_bound, 
                                velocity_lower_bound = args.velocity_lower_bound,
                                velocity_redundant_num=args.cost_1_redundant_num,
                                velocity_lower_redundant_num = args.cost_2_redundant_num,
                               collision_redundant_num=args.cost_0_redundant_num)
    else:
        demo_env = gym.make(args.task)

    agent = SACLagAgent(
        env=demo_env,
        logger=logger,
        # general task params
        device=args.device,
        thread=args.thread,
        seed=args.seed,
        # algorithm params
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        hidden_sizes=args.hidden_sizes,
        auto_alpha=args.auto_alpha,
        alpha_lr=args.alpha_lr,
        alpha=args.alpha,
        tau=args.tau,
        n_step=args.n_step,
        # Lagrangian specific arguments
        use_lagrangian=args.use_lagrangian,
        lagrangian_pid=args.lagrangian_pid,
        cost_limit=args.cost_limit,
        rescaling=args.rescaling,
        # Base policy common arguments
        gamma=args.gamma,
        conditioned_sigma=args.conditioned_sigma,
        unbounded=args.unbounded,
        last_layer_scale=args.last_layer_scale,
        deterministic_eval=args.deterministic_eval,
        action_scaling=args.action_scaling,
        action_bound_method=args.action_bound_method,
        lr_scheduler=None,
        vanilla=args.vanilla,
        random_con_selection=args.random_con_selection,
        gradient_shaping=args.gradient_shaping,
        max_singleAC = args.max_singleAC,
    )

    training_num = min(args.training_num, args.episode_per_collect)
    worker = eval(args.worker)
    if args.safety_gymnasium_task:
        train_envs = worker([lambda: vel_wrapper(gym.make(args.task), 
                                                 velocity_bound=args.velocity_bound, 
                                                 velocity_lower_bound = args.velocity_lower_bound,
                                                 velocity_redundant_num=args.cost_1_redundant_num,
                                                 velocity_lower_redundant_num = args.cost_2_redundant_num,
                                                 collision_redundant_num=args.cost_0_redundant_num) for _ in range(training_num)])
        
        test_envs = worker([lambda: vel_wrapper(gym.make(args.task), 
                                                velocity_bound=args.velocity_bound, 
                                                velocity_lower_bound = args.velocity_lower_bound,
                                                velocity_redundant_num=args.cost_1_redundant_num,
                                                velocity_lower_redundant_num = args.cost_2_redundant_num,
                                                collision_redundant_num=args.cost_0_redundant_num) for _ in range(args.testing_num)])
    else:
        train_envs = worker([lambda: redundant_wrapper(gym.make(args.task), 
                                                       cost_0_redunt_num=args.cost_0_redundant_num, 
                                                       cost_1_redunt_num=args.cost_1_redundant_num,
                                                       cost_2_redunt_num=args.cost_2_redundant_num) for _ in range(training_num)])
        
        test_envs = worker([lambda: redundant_wrapper(gym.make(args.task), 
                                                      cost_0_redunt_num=args.cost_0_redundant_num, 
                                                      cost_1_redunt_num=args.cost_1_redundant_num,
                                                      cost_2_redunt_num=args.cost_2_redundant_num) for _ in range(args.testing_num)])
    # start training
    agent.learn(
        train_envs=train_envs,
        test_envs=test_envs,
        epoch=args.epoch,
        episode_per_collect=args.episode_per_collect,
        step_per_epoch=args.step_per_epoch,
        update_per_step=args.update_per_step,
        buffer_size=args.buffer_size,
        testing_num=args.testing_num,
        batch_size=args.batch_size,
        reward_threshold=args.reward_threshold,  # for early stop purpose
        save_interval=args.save_interval,
        resume=args.resume,
        save_ckpt=args.save_ckpt,
        verbose=args.verbose,
    )

    if __name__ == "__main__":
        # Let's watch its performance!
        from fsrl.data import FastCollector
        env = gym.make(args.task)
        agent.policy.eval()
        collector = FastCollector(agent.policy, env)
        result = collector.collect(n_episode=10, render=args.render)
        rews, lens, cost = result["rew"], result["len"], result["cost"]
        print(f"Final eval reward: {rews.mean()}, cost: {cost}, length: {lens.mean()}")

        agent.policy.train()
        collector = FastCollector(agent.policy, env)
        result = collector.collect(n_episode=10, render=args.render)
        rews, lens, cost = result["rew"], result["len"], result["cost"]
        print(f"Final train reward: {rews.mean()}, cost: {cost}, length: {lens.mean()}")


if __name__ == "__main__":
    train()
