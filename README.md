# Gradient Shaping for Multi-constraint Safe Reinforcement Learning
Implementation code for L4DC 2024 paper https://arxiv.org/html/2312.15127.

## Installation

This repo can be installed on Ubuntu 18.04. Create a conda environment (recommended Python 3.8), then install [PyTorch](https://pytorch.org/get-started/locally/) based on your platform. Then run
```
cd GradS
pip install -e .
```

To use the `Bullet-Safety-Gym` environment, please use the repo in this folder which supports `gym>=0.26.0` API and multiple safety constraints:

```
cd ../Bullet-Safety-Gym
pip install -e .
```

## Structure
The structure of this repo is as follows:
```
├── GradS  # implementation of MC safe RL algorithms
│   ├── fsrl
│   │   ├── data # data collector
│   │   ├── agent # safe RL agent
│   │   ├── policy # safe RL policy
│   │   ├── trainer # safe RL trainer
│   │   ├── config # training config
│   │   ├── utils # logger and network 
│   ├── examples # training files
├── Bullet-Safety-Gym  # training envs
```

## Usage
By running the following code in the terminal, you will be able to test the GradS with PPO-Lag as a base algorithm.
```
cd GradS/examples
python train_ppo_agent.py --seed [SS] --short_task [TT]
```
where SS specifies the random seed and short_task specifies the task. You may choose it from \{"BC-v2", "CC-v2", "DC-v2", "BC-v3", "CC-v3", "DC-v3"\}.
By default, the seed is set to be 10, and cost limits for different constraints in each task can be found in the file "GradS/fsrl/config/ppol_cfg.py". After running this command, you can check the training stats for the "BC-v3" task in the wandb logger. Under the tabs "train" and "test", we provide the "sub_cost_0", "sub_cost_1", and "sub_cost_2", which represents the high-speed cost, boundary cost, and low-speed cost, respectively. 

## Acknowledge
This repo is partly based on [Tianshou](https://github.com/thu-ml/tianshou) and [FSRL](https://github.com/liuzuxin/FSRL).


<!-- ## Contributing

The main maintainer of this project is: [Yihang Yao](yihangya@andrew.cmu.edu) -->

