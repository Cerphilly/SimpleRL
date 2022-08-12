![Python Depend](https://img.shields.io/badge/Python-3.6-blue) ![TF Depend](https://img.shields.io/badge/TensorFlow-2.6-orange) ![GYM Depend](https://img.shields.io/badge/openai%2Fgym-0.17.3-green)

# SimpleRL
Personal Reinforcement Learning (RL) repo made to backup codes I implemented

# What is this?
**SimpleRL** is a repository that contains variety of Deep Reinforcement Learning (Deep RL) algorithms using Tensorflow2. This repo is mainly made to backup codes that I implemented while studying RL, but also made to let others easily learn Deep RL. For easy-to-understand RL code, each algorithm is written as simple as possible. This repository will be constantly updated with new Deep RL algorithms.   

# Algorithms
- **DQNs**<br>
  - [DQN](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
  - [DDQN](https://arxiv.org/pdf/1509.06461.pdf)
  - [Dueling DQN](https://arxiv.org/pdf/1511.06581.pdf)
- [DDPG](https://arxiv.org/pdf/1509.02971.pdf)
- [TD3](https://arxiv.org/pdf/1802.09477.pdf)
- [SAC_v1](https://arxiv.org/pdf/1801.01290.pdf)
- [SAC_v2](https://arxiv.org/pdf/1812.05905.pdf)
- [REINFORCE](https://link.springer.com/content/pdf/10.1007/BF00992696.pdf)
- [VPG](https://spinningup.openai.com/en/latest/algorithms/vpg.html)
- [TRPO](https://arxiv.org/pdf/1502.05477.pdf)
- [PPO](https://arxiv.org/pdf/1707.06347.pdf)
- [D2RL](https://arxiv.org/pdf/2010.09163.pdf)
- **ImageRL**<br>
  - [CURL](https://arxiv.org/pdf/2004.04136.pdf)
  - [RAD](https://arxiv.org/pdf/2004.14990.pdf)
  - [SAC_AE](https://arxiv.org/pdf/1910.01741.pdf)
  - [DBC](https://arxiv.org/pdf/2006.10742.pdf)
  
# Installation
This code is built in Windows using Anaconda. You can see full environment exported as yaml file (tf2.yaml)   

# How to use
You can run algorithms by using examples in `SimpleRL/Example` folder. All `run_XXX.py` defines **Hyperparameters** for the experiment.
Also, RL `Environment` and `Algorithm`, and its `Trainer` is required to run the experiment.

# Warning
There are some unresovled errors and things you have to know:
1. Official benchmark score may not be guaranteed. This can happen due to random seed, hyperparameter, etc.
2. Especially, **On-policy algorithms** (REINFORCE, VPG, TRPO, PPO) in continous action environment shows poor performance for unknown reasons. 
3. DBC (Deep Bisimulation for Control) also seems to show performance poorer than the official paper.      


# Reference
- https://spinningup.openai.com/en/latest/index.html
- https://github.com/keiohta/tf2rl
- https://github.com/reinforcement-learning-kr/pg_travel
- https://github.com/MishaLaskin/rad
- https://github.com/MishaLaskin/curl
- https://github.com/denisyarats/pytorch_sac_ae
- https://github.com/facebookresearch/deep_bisim4control
