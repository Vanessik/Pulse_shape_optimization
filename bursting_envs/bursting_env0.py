import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
from gym import Env
from gym.spaces import Discrete, MultiDiscrete, MultiBinary, Box, ImpulseBox
import sys
sys.path.append('/Pulse_optimization/bursting_sources')
import oscillator_bursting as oscillator_cpp
import random

class BurstingEnv0D(gym.Env):
    def __init__(self, cfg):
        super(BurstingEnv0D, self).__init__()
        self.cfg = cfg
#         system parameters
        self.nosc = cfg['n_nodes']
        self.epsilon = cfg['epsilon']
        self.frrms = cfg['frrms']
        self.ndim = cfg['ndim']
        self.impulse = cfg['impulse']
        self.autonom_time = cfg['autonom_time']
        self.dt = cfg['dt']
        self.T = 150
        self.ep_length = cfg['ep_length'] #1 for debug
        self.hist_length = cfg['hist_length']
        self.n_pulses = cfg['n_pulses']
        self.n_trans_pulses = cfg['n_trans_pulses']
        self.ndim = cfg['ndim']
        self.ampl_reward = cfg['ampl_reward']
        self.action_space = Box(-1, 1, shape=(1, ), dtype=np.float32) # Use normalized actions  
        self.space_type = cfg['state_space_type']
            
        if self.space_type == 'discrete':
            print('Discrete state space!')
            self.observation_space = Discrete(self.ep_length) #2pi/self.ep_length*i
        else:
            print('Box state space!')
            self.observation_space = Box(low=0, high=2*np.pi, shape=(1, ), dtype=np.float32)
        self.skip_steps = 0
        # Create sync state
        self.done = False
        # Reset environment
        self.state = self.reset()
        
    def denormalize_actions(self, action):
        # x = (y - y1) / (y2 - y1) * (x2 - x1) + x1 linear interpolation
        de_act = self.low_a + (self.high_a - self.low_a) * (action + 1) / 2 
        return de_act

    def step(self, action):
        width_p  = round(0.1 *self.T/ self.dt) #fixed
        kfactor = 0
        gap = 0
        theta = np.array([2 * np.pi / self.ep_length * self.current_step])
        if self.space_type == 'discrete':
            theta = theta[0]       
        last_idx = oscillator_cpp.Return_end_idx(self.history, self.hist_length) - 1
        self.std0 = oscillator_cpp.Calc_std(self.history, 1, last_idx)
        pos_imp = float(action[0])
        neg_imp = -pos_imp / kfactor if kfactor != 0 else 0
        for i in range(self.n_trans_pulses):
            self.y = oscillator_cpp.Phase_state(float(theta), self.y, self.history, self.phase_oscillator, self.hist_length)
            self.y = oscillator_cpp.Make_biphasic_step(self.y, self.history, self.phase_oscillator, width_p, gap, round(kfactor*width_p), pos_imp, neg_imp, self.skip_steps, self.hist_length)
        start_idx = oscillator_cpp.Return_end_idx(self.history, self.hist_length) 
        for i in range(self.n_pulses):
            self.y = oscillator_cpp.Phase_state(float(theta), self.y, self.history, self.phase_oscillator, self.hist_length)
            self.y = oscillator_cpp.Make_biphasic_step(self.y, self.history, self.phase_oscillator, width_p, gap, round(kfactor*width_p), pos_imp, neg_imp, self.skip_steps, self.hist_length)
        
        end_idx = oscillator_cpp.Return_end_idx(self.history, self.hist_length) - 1      
        self.std = oscillator_cpp.Calc_std(self.history, start_idx, end_idx)

        self.current_step += 1
        self.done = self.current_step >= self.ep_length
        
        oscillator_cpp.free_vector(self.history, 1, self.hist_length)
        oscillator_cpp.free_vector(self.y, 1, self.ndim*self.nosc)
        oscillator_cpp.free_vector(self.phase_oscillator, 0, self.ndim*self.hist_length)
        del self.phase_oscillator
        del self.y
        del self.history
        # Reset environment
        self.reset2()
        
        return theta, self.reward(pos_imp), self.done, {} 
    
    def reset2(self):
        self.y = oscillator_cpp.init(self.nosc, self.epsilon, self.frrms, self.ndim)
        self.history = oscillator_cpp.init_history(self.hist_length)
        self.phase_oscillator = oscillator_cpp.init_phase_oscillator(self.hist_length)
        theta = random.sample([2*np.pi/self.ep_length*i for i in range(self.ep_length)], k=1)
        self.y = oscillator_cpp.Make_step3(self.y, self.history, self.phase_oscillator, 1000, self.hist_length)
        self.y = oscillator_cpp.Phase_state(float(np.array([theta])), self.y, self.history, self.phase_oscillator, self.hist_length)        
    
    def reset(self):
        self.done = False
        self.current_step = 0 
        self.y = oscillator_cpp.init(self.nosc, self.epsilon, self.frrms, self.ndim)
        self.history = oscillator_cpp.init_history(self.hist_length)
        self.phase_oscillator = oscillator_cpp.init_phase_oscillator(self.hist_length)
        theta = random.sample([2*np.pi/self.ep_length*i for i in range(self.ep_length)], k=1)
        self.y = oscillator_cpp.Make_step3(self.y, self.history, self.phase_oscillator, 1000, self.hist_length)
        self.y = oscillator_cpp.Phase_state(float(np.array([theta])), self.y, self.history, self.phase_oscillator, self.hist_length)
        if self.space_type == 'discrete':
            return np.array(theta)[0]
        return np.array(theta)

    def reward(self, amplitude):
        rews = self.std0 / self.std - self.ampl_reward * abs(amplitude)
        return rews