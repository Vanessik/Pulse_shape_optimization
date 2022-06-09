import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
from gym import Env
from collections import deque
from gym.spaces import Discrete, MultiDiscrete, MultiBinary, Box, ImpulseBox
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('/Pulse_optimization/kuramoto_sources')
import oscillator_cpp


class KuramotoEnv2D(gym.Env):
    def __init__(self, config):
        print('in kuramoto')
        super(KuramotoEnv2D, self).__init__()
        self.cfg = config
        self.T = 4 * np.pi / (config['slow_freq'] + config['fast_freq']) # Period of oscillations in system
        self.dt = config['dt']
        self.impulse = config['impulse']
        # Action space
        self.low_a = np.array([self.dt ]) 
        self.high_a = np.array([0.01 * self.T ])
        self.low_s = 0 
        self.high_s = 2 * np.pi
        # Define action and observation spaces
        self.action_space = Box(-1, 1, shape=(1, ), dtype=np.float32) # Use normalized actions
        self.observation_space = Box(low=self.low_s, high=self.high_s, shape=(1, ), dtype=np.float32)
        self.done = False
        self.ep_length = config['ep_length']
        self.shift = config['shift']
        # Init states
        self.phi = oscillator_cpp.init(config['nptrans'])
        self.sync_phi = oscillator_cpp.init_state_copy(self.phi) #fixed
        self.sync_sync_phi = oscillator_cpp.init_state_copy(self.phi) #fixed
        self.y = oscillator_cpp.init_MF() 
        # Reset environment 
        self.reset()

    def denormalize_actions(self, action):
        # x = (y - y1) / (y2 - y1) * (x2 - x1) + x1 linear interpolation
        de_act = self.low_a + (self.high_a - self.low_a) * (action + 1) / 2 
        return de_act

    def reset(self):
        # Return to sync state
        self.current_step = 0 
        self.done = False
        oscillator_cpp.copy_vecs(self.phi, self.sync_phi)
        oscillator_cpp.MeanField(self.phi, self.y)
        self.state = oscillator_cpp.Calc_Theta(self.y)
        return np.array([self.state])

    def step(self, action):
        width_p =  round(self.denormalize_actions(action)[0] / self.dt)
        # Calculate mean field
        oscillator_cpp.copy_vecs(self.sync_sync_phi, self.phi)   
        oscillator_cpp.MeanField(self.phi, self.y)
        R0, theta0 = oscillator_cpp.Calc_R(self.y), oscillator_cpp.Calc_Theta(self.y)
        self.state = theta0
        oscillator_cpp.Make_step(self.phi, self.y, width_p, 0, 0, self.impulse, 0., 0)
        oscillator_cpp.MeanField(self.phi, self.y)
        R, theta = oscillator_cpp.Calc_R(self.y), oscillator_cpp.Calc_Theta(self.y)
        oscillator_cpp.copy_vecs(self.phi, self.sync_sync_phi) 
        oscillator_cpp.Make_step(self.phi, self.y, 0, 0, 0, self.impulse, 0., self.shift)
        r = self.reward(R0, R)
        self.current_step += 1
        self.done = self.current_step >= self.ep_length
        return np.array([self.state]), r, self.done, {} 

    def reward(self, R0, R):
        return 5*(R0-R)/np.abs(self.impulse) # Want maximize R0 - R

    
class KuramotoEnv3D(gym.Env):
    def __init__(self, config, model=None):
        self.cfg = config
        self.T = 4 * np.pi / (config['slow_freq'] + config['fast_freq']) # Period of oscillations in system
        self.dt = config['dt']
        self.impulse = config['impulse']

        self.low_a = np.array([0.001, 0, 0.5 ]) #change low from self.dt/T
        self.high_a = np.array([0.01 * self.T , 0.97 * self.T, 5.]) 
        self.low_s = 0 * np.pi
        self.high_s = 2 * np.pi
        # Define action and observation spaces
        self.action_space = ImpulseBox(-1, 1, [self.low_a, self.high_a], shape=(3,), dtype=np.float32)
        self.observation_space = Box(low=self.low_s, high=self.high_s, shape=(1, ), dtype=np.float32)
        self.done = False
        self.avg = config['avg']
        self.ep_length = config['ep_length']
        self.shift = config['shift']
         # Init states
        print(config['nptrans'])
        self.phi = oscillator_cpp.init(config['nptrans'])
        self.sync_phi = oscillator_cpp.init_state_copy(self.phi) #fixed
        self.sync_sync_phi = oscillator_cpp.init_state_copy(self.phi) #fixed
        self.y = oscillator_cpp.init_MF() 
        oscillator_cpp.MeanField(self.phi, self.y)
        R0, theta0 = oscillator_cpp.Calc_R(self.y), oscillator_cpp.Calc_Theta(self.y)
        print('after sync: ', R0, theta0)
        # Reset environment 
        self.reset()
       
   
    def denormalize_actions(self, action):
        # x = (y - y1) / (y2 - y1) * (x2 - x1) + x1 linear interpolation
        de_act = self.low_a + (self.high_a - self.low_a) * (action + 1) / 2 
        return de_act

    def reset(self):
        # Return to sync state
        self.current_step = 0 
        self.done = False
        oscillator_cpp.copy_vecs(self.phi, self.sync_phi)
        oscillator_cpp.MeanField(self.phi, self.y)
        self.state = oscillator_cpp.Calc_Theta(self.y)
        return np.array([self.state])
    
    def step(self, action):   
        width_p, gap, kfactor = round(self.denormalize_actions(action)[0] / self.dt), round(self.denormalize_actions(action)[1]/ self.dt), self.denormalize_actions(action)[2]
        # Calculate mean field
        oscillator_cpp.copy_vecs(self.sync_sync_phi, self.phi)    #new_sync
        oscillator_cpp.MeanField(self.phi, self.y)
        R0, theta0 = oscillator_cpp.Calc_R(self.y), oscillator_cpp.Calc_Theta(self.y)
        self.state = theta0
        
        pos_imp = self.impulse
        neg_imp = -self.impulse/kfactor if kfactor != 0 else 0
        R_avg, theta_avg = 0, 0
        for i in range(self.avg):
            oscillator_cpp.copy_vecs(self.phi, self.sync_sync_phi)
            oscillator_cpp.Make_step(self.phi, self.y, width_p, gap, round(kfactor*width_p), pos_imp, neg_imp, 0)
            oscillator_cpp.MeanField(self.phi, self.y)
            R, theta = oscillator_cpp.Calc_R(self.y), oscillator_cpp.Calc_Theta(self.y)
            R_avg += R
            theta_avg += theta 
        R, theta = R_avg / self.avg, theta_avg / self.avg
        
        oscillator_cpp.copy_vecs(self.phi, self.sync_sync_phi) 
        oscillator_cpp.Make_step(self.phi, self.y, 0, 0, 0, self.impulse, 0., self.shift)
            
        r = self.reward(R0, R)
        self.current_step += 1
        self.done = self.current_step >= self.ep_length
        return np.array([self.state]), r, self.done, {} 
    
    def reward(self, R0, R):
        return 5*(R0 - R) / np.abs(self.impulse)   # Want maximize R0 - R