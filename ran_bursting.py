import gym
from gym import error, spaces, utils
from gym.utils import seeding
from stable_baselines.common import set_global_seeds
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv,SubprocVecEnv,VecNormalize, VecEnv
from stable_baselines.common.callbacks import CheckpointCallback
from stable_baselines import PPO2, ACKTR
import os
import numpy as np
import yaml
from bursting_envs.bursting_env3 import  BurstingEnv3D
from collections import deque
import matplotlib.pyplot as plt
from stable_baselines.common import make_vec_env


if __name__ == "__main__":
    with open('./configs/bursting_cfg.yaml', 'r') as ymlfile:
        cfg = yaml.load(ymlfile,Loader=yaml.FullLoader)

    if cfg['num_cpu'] == 1:
        env = DummyVecEnv([lambda: BurstingEnv3D(cfg)])
    else:
        env = SubprocVecEnv([lambda: BurstingEnv3D(cfg) for i in range(cfg['num_cpu'])])
    env.reset()
    checkpoint_callback = CheckpointCallback(save_freq=cfg['freq'], save_path=cfg['save_path'],                           name_prefix=cfg['prefix'])
    model = ACKTR(MlpPolicy, env, verbose=1,  tensorboard_log=cfg['tensorboard_log'])
    model.learn(cfg['num_learning_steps'], log_interval=cfg['log_interval'], callback=checkpoint_callback)
    model.save(cfg['save_model'])
