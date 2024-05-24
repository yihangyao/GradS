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

from fsrl.agent import PPOLagAgent
from fsrl.config.ppol_cfg import (
    BC_v2,
    CC_v2,
    DC_v2,
    BC_v3,
    CC_v3,
    DC_v3,
    TrainCfg,
)

from fsrl.utils import BaseLogger, TensorboardLogger, WandbLogger
from fsrl.utils.exp_util import auto_name, redundant_wrapper, vel_wrapper

TASK_TO_CFG = {
    # bullet safety gym tasks
    "BC-v2": BC_v2,
    "CC-v2": CC_v2,
    "DC-v2": DC_v2,
    "BC-v3": BC_v3,
    "CC-v3": CC_v3,
    "DC-v3": DC_v3,
}


@pyrallis.wrap()
def train(args: TrainCfg):

    task = args.task
    short_task = args.short_task
    default_cfg = TASK_TO_CFG[short_task]() if short_task in TASK_TO_CFG else TrainCfg()
    # use the default configs instead of the input args.
    if args.use_default_cfg:
        default_cfg.seed = args.seed
        default_cfg.device = args.device
        default_cfg.logdir = args.logdir
        default_cfg.project = default_cfg.short_task
        default_cfg.group = args.group
        default_cfg.suffix = args.suffix
        args = default_cfg

    cost_0_thres = args.thre_0
    cost_1_thres = args.thre_1
    cost_2_thres = args.thre_2

    # cost_limit_new = [cost_0_thres, cost_1_thres, cost_2_thres]
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
    # print(args.cost_limit)
    # raise RuntimeError

    # setup logger
    cfg = asdict(args)
    default_cfg = asdict(default_cfg)
    if args.name is None:
        args.name = "GradS(PPOL)-" + auto_name(default_cfg, cfg, args.prefix, args.suffix)
    if args.group is None:
        args.group = "GradS-" + args.short_task # + "-cost-0-" + str(cost_0_thres) + "-cost-1-" + str(cost_1_thres) + "-cost-2-" + str(cost_2_thres) # + \
        if args.cost_0_redundant_num > 0:
            args.group += ("-cost-0-" + str(cost_0_thres))
        if args.cost_1_redundant_num > 0:
            args.group += ("-cost-1-" + str(cost_1_thres))
        if args.cost_2_redundant_num > 0:
            args.group += ("-cost-2-" + str(cost_1_thres))
        args.group += args.suffix
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

    agent = PPOLagAgent(
        env=demo_env,
        logger=logger,
        device=args.device,
        thread=args.thread,
        seed=args.seed,
        lr=args.lr,
        hidden_sizes=args.hidden_sizes,
        unbounded=args.unbounded,
        last_layer_scale=args.last_layer_scale,
        target_kl=args.target_kl,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        gae_lambda=args.gae_lambda,
        eps_clip=args.eps_clip,
        dual_clip=args.dual_clip,
        value_clip=args.value_clip,
        advantage_normalization=args.norm_adv,
        recompute_advantage=args.recompute_adv,
        use_lagrangian=args.use_lagrangian,
        lagrangian_pid=args.lagrangian_pid,
        cost_limit=args.cost_limit,
        rescaling=args.rescaling,
        gamma=args.gamma,
        max_batchsize=args.max_batchsize,
        reward_normalization=args.rew_norm,
        deterministic_eval=args.deterministic_eval,
        action_scaling=args.action_scaling,
        action_bound_method=args.action_bound_method,
        constraint_num=len(args.cost_limit),
        vanilla=args.vanilla,
        random_con_selection=args.random_con_selection,
        gradient_shaping=args.gradient_shaping,
        max_singleAC = args.max_singleAC,
        conditioned_sigma = args.conditioned_sigma
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

    valid_cost_num = len(cost_limit_new)
    
    # start training
    agent.learn(
        train_envs=train_envs,
        test_envs=test_envs,
        epoch=args.epoch,
        episode_per_collect=args.episode_per_collect,
        step_per_epoch=args.step_per_epoch,
        repeat_per_collect=args.repeat_per_collect,
        buffer_size=args.buffer_size,
        testing_num=args.testing_num,
        batch_size=args.batch_size,
        reward_threshold=args.reward_threshold,
        save_interval=args.save_interval,
        resume=args.resume,
        save_ckpt=args.save_ckpt,
        verbose=args.verbose,
        valid_cost_num=valid_cost_num
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
