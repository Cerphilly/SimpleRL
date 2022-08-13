![Python Depend](https://img.shields.io/badge/Python-3.6-blue) ![TF Depend](https://img.shields.io/badge/TensorFlow-2.6-orange) ![GYM Depend](https://img.shields.io/badge/openai%2Fgym-0.17.3-green)

# SimpleRL
Personal Reinforcement Learning (RL) repo made to backup codes I implemented

# What is this?
**SimpleRL** is a repository that contains variety of Deep Reinforcement Learning (Deep RL) algorithms using Tensorflow2. This repo is mainly made to backup codes that I implemented while studying RL, but also made to let others easily learn Deep RL. For easy-to-understand RL code, each algorithm is written as simple as possible. This repository will be constantly updated with new Deep RL algorithms.   

# Algorithms
- **DQNs**<br>
  - [DQN](#dqn)
  - [DDQN](#ddqn)
  - [Dueling DQN](#duelingdqn)
- [DDPG](#ddpg)
- [TD3](#td3)
- [SAC_v1](#sacv1)
- [SAC_v2](#sacv2)
- [REINFORCE](#reinforce)
- [VPG](#vpg)
- [TRPO](#trpo)
- [PPO](#ppo)
- **ImageRL**<br>
  - [CURL](#curl)
  - [RAD](#rad)
  - [SAC_AE](#sacae)
  - [DBC](#dbc)
- [D2RL](#d2rl)


## DQNs
Deep Q Network (DQN) and algorithms derived from it.

<a name='dqn'></a>
### DQN (Deep Q-Networks)
**[Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf), Mnih et al, 2013** 

<a name='ddqn'></a>
### DDQN (Double Deep Q-Networks)
**[Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/pdf/1509.06461.pdf), Hasselt et al 2015.**

<a name='duelingdqn'></a>
### Dueling DQN (Dueling Deep Q-Networks)
**[Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/pdf/1511.06581.pdf), Wang et al, 2015.**

<hr>

<a name='ddpg'></a>
### DDPG (Deep Deterministic Policy Gradient)
**[Continuous Control With Deep Reinforcement Learning](https://arxiv.org/pdf/1509.02971.pdf), Lillicrap et al, 2015.**

<a name='td3'></a>
### TD3 (Twin Delayed Deep Deterministic Policy Gradient)
**[Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/pdf/1802.09477.pdf), Fujimoto et al, 2018.**

<a name='sacv1'></a>
### SAC_v1 (Soft Actor Critics)
**[Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/pdf/1801.01290.pdf), Haarnoja et al, 2018.**

<a name='sacv2'></a>
### SAC_v2 (Soft Actor Critics)
**[Soft actor-critic algorithms and applications](https://arxiv.org/pdf/1812.05905.pdf), Haarnoja et al, 2018.**

<hr>

<a name='reinforce'></a>
### REINFORCE 
**[Simple statistical gradient-following algorithms for connectionist reinforcement learning](https://link.springer.com/content/pdf/10.1007/BF00992696.pdf), Ronald J. Williams, 1992.**

<a name='vpg'></a>
### VPG (Vanilla Policy Gradient)
**[Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf), Sutton et al, 2000.**

<a name='trpo'></a>
### TRPO (Trust Region Policy Optimization)
**[Trust Region Policy Optimization](https://arxiv.org/pdf/1502.05477.pdf), Schulman et al, 2015.**

<a name='ppo'></a>
### PPO (Proximal Policy Optimization)
**[Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347.pdf), Schulman et al, 2017.**

## ImageRL
RL algorithms that learns policy from pixels

<a name='curl'></a>
### CURL (Contrastive Unsupervised Reinforcement Learning)
**[CURL: Contrastive Unsupervised Representation Learning for Sample-Efficient Reinforcement Learning](https://arxiv.org/pdf/2004.04136.pdf), Srinivas et al, 2020.**

<a name='rad'></a>
### RAD (Reinforcement learning with Augmented Data)
**[RAD: Reinforcement Learning with Augmented Data](https://arxiv.org/pdf/2004.14990.pdf), Laskin et al, 2020.**

<a name='sacae'></a>
### SAC_AE (Soft Actor Critics with AutoEncoder)
**[Improving Sample Efficiency in Model-Free Reinforcement Learning from Images](https://arxiv.org/pdf/1910.01741.pdf), Yarats et al, 2020.**

<a name='dbc'></a>
### DBC (Deep Bisimulation for Control)
**[Learning Invariant Representations for Reinforcement Learning without Reconstruction](https://arxiv.org/pdf/2006.10742.pdf), A. Zhang et al, 2020.**

<hr>

<a name='d2rl'></a>
### D2RL (Deep Dense Architectures in Reinforcement Learning)
**[D2RL: Deep Dense Architectures in Reinforcement Learning](https://arxiv.org/pdf/2010.09163.pdf), Sinha et al, 2020**

<hr>

# Installation
This code is built in Windows using Anaconda. You can see full environment exported as yaml file (tf2.yaml)   

# How to use
You can run algorithms by using examples in `SimpleRL/Example` folder. All `run_XXX.py` defines **Hyperparameters** for the experiment.
Also, RL `Environment` and `Algorithm`, and its `Trainer` is required to run the experiment.

# Warning
There are some unresovled errors and issues you have to know:
1. Official benchmark score may not be guaranteed. This can happen due to random seed, hyperparameter, etc.
2. Especially, **On-policy algorithms** (REINFORCE, VPG, TRPO, PPO) in continous action environment shows poor performance for unknown reasons. 
3. DBC (Deep Bisimulation for Control) also seems to show performance poorer than the official paper.      

***Any advice on code is always welcomed!***

# Reference
- https://spinningup.openai.com/en/latest/index.html
- https://github.com/keiohta/tf2rl
- https://github.com/reinforcement-learning-kr/pg_travel
- https://github.com/MishaLaskin/rad
- https://github.com/MishaLaskin/curl
- https://github.com/denisyarats/pytorch_sac_ae
- https://github.com/facebookresearch/deep_bisim4control
