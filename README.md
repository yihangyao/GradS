# Gradient Shaping for Multi-constraint Safe Reinforcement Learning
Supplementary code for L4DC 2024 submission. Do not distribute.

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
│   ├── fsrl # core fold
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
By running the following code in the terminal, you will be able to test the GradS algorithm with PPO-Lag using random seed [XX].
```
cd GradS/examples
python train_ppo_agent.py --seed [XX]
```
By default, the seed is set to be 10. You may also try other seeds such as 20, 30, 40, and 50. Then you can check the training stats for the "BC-v3" task in the wandb logger. Under the tabs "train" and "test", we provide the "sub_cost_0", "sub_cost_1", and "sub_cost_2", which represents the high-speed cost, boundary cost, and low-speed cost, respectively. The corresponding thresholds are set to be 50, 20, and 40 by default.

<!-- ## Contributing

The main maintainer of this project is: [Yihang Yao](yihangya@andrew.cmu.edu) -->

