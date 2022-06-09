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

class BurstingEnv1D(gym.Env):
    def __init__(self):
        super(BurstingEnv1D, self).__init__()
#         self.cfg = cfg
#         system parameters
        self.nosc = 1000#cfg['n_nodes']
        self.epsilon = 0.2 #cfg['epsilon']
        self.frrms = 0.2 #cfg['frrms']
        self.ndim = 3#cfg['ndim']
        self.impulse = 0.5#cfg['impulse']
        self.autonom_time = 10000#cfg['autonom_time']
        self.dt = 0.1#cfg['dt']
        self.T = 150
        self.ep_length = 10#cfg['ep_length'] #1 for debug
        self.hist_length = 120000#cfg['hist_length']
        self.n_pulses = 10 #cfg['n_pulses']
        self.n_trans_pulses = 50#cfg['n_trans_pulses']
        self.ndim = 3 #cfg['ndim']
        self.low_a = np.array([0.01*self.T]) #1 step
        self.high_a = np.array([0.06 * self.T])  #6%
        self.low_s = 0.
        self.high_s = 3.14 / 2
        self.optimize_amplitude = False #cfg['optimize_amplitude']
        self.ampl_reward = 0. # cfg['ampl_reward']
        
        if self.optimize_amplitude:
            print('Predict width of first pulse and amplitude!')
            self.action_space = Box(-1, 1, shape=(2, ), dtype=np.float32) # Use normalized actions
        else:
            print('Predict width of first pulse!')
            self.action_space = Box(-1, 1, shape=(1, ), dtype=np.float32) # Use normalized actions
            
        self.space_type = 'not'#, cfg['state_space_type']
            
        if self.space_type == 'discrete':
            print('Discrete state space!')
            self.observation_space = Discrete(self.ep_length) #2pi/self.ep_length*i
        else:
            print('Box state space!')
            self.observation_space = Box(low=self.low_s, high=self.high_s, shape=(1, ), dtype=np.float32)
        self.skip_steps = 0
        # Create sync state
        self.done = False
        # Init state, history and phase_oscillator
        self.y = oscillator_cpp.init(self.nosc, self.epsilon, self.frrms, self.ndim)
        self.history = oscillator_cpp.init_history(self.hist_length)
        self.phase_oscillator = oscillator_cpp.init_phase_oscillator(self.hist_length)
        # Reset environment
        self.state = self.reset()
        
    def denormalize_actions(self, action):
        # x = (y - y1) / (y2 - y1) * (x2 - x1) + x1 linear interpolation
        de_act = self.low_a + (self.high_a - self.low_a) * (action + 1) / 2 
        return de_act
    def get_history(self):
        history_arr = []
        end_idx = oscillator_cpp.Return_end_idx(self.history, self.hist_length)
        for i in range(1, end_idx, 1):
            history_arr.append(oscillator_cpp.Show_history(self.history, i))
        return history_arr

    def step(self, action):
        width_p  = round(self.denormalize_actions(action)[0] / self.dt)
        kfactor = 0
        gap = 0
        theta = np.array([self.low_s + (self.high_s - self.low_s) / self.ep_length * self.current_step])
        print('theta is ', theta)
        if self.space_type == 'discrete':
            theta = theta[0]       
        last_idx = oscillator_cpp.Return_end_idx(self.history, self.hist_length) - 1
        self.std0 = oscillator_cpp.Calc_std(self.history, 1, last_idx)
#         print('std0', self.std0, last_idx)
        if self.ampl_reward:
            pos_imp = float(action[1])
        else:
            pos_imp = self.impulse
        neg_imp = -pos_imp / kfactor if kfactor != 0 else 0
        for i in range(self.n_trans_pulses):
#             print('in trans',oscillator_cpp.Return_end_idx(self.history, self.hist_length))
            oscillator_cpp.Phase_state(float(theta), self.y, self.history, self.phase_oscillator, self.hist_length)
#             print('in trans after phase estim',oscillator_cpp.Return_end_idx(self.history, self.hist_length))
            if float(theta)>=0 and float(theta)<3.14:
                pos_imp = -np.abs(pos_imp)
            else:
                pos_imp = np.abs(pos_imp)
#             print('theta', float(theta), 'imp', pos_imp)
            oscillator_cpp.Make_biphasic_step(self.y, self.history, self.phase_oscillator, width_p, gap, round(kfactor*width_p), pos_imp, neg_imp, self.skip_steps, self.hist_length)
            theta_ = np.mod(float(theta)+3.14, 6.28)
#             print('polar theta', theta_)
            oscillator_cpp.Phase_state(float(theta_), self.y, self.history, self.phase_oscillator, self.hist_length)
            oscillator_cpp.Make_biphasic_step(self.y, self.history, self.phase_oscillator, width_p, gap, round(kfactor*width_p), -pos_imp, neg_imp, self.skip_steps, self.hist_length)
#             print('in trans after bipphase estim',oscillator_cpp.Return_end_idx(self.history, self.hist_length))
        start_idx = oscillator_cpp.Return_end_idx(self.history, self.hist_length) 
        for i in range(self.n_pulses):
            oscillator_cpp.Phase_state(float(theta), self.y, self.history, self.phase_oscillator, self.hist_length)
            if float(theta)>=0 and float(theta)<3.14:
                pos_imp = -np.abs(pos_imp)
            else:
                pos_imp = np.abs(pos_imp)
#             print('cont theta', float(theta), 'imp', pos_imp)
            oscillator_cpp.Make_biphasic_step(self.y, self.history, self.phase_oscillator, width_p, gap, round(kfactor*width_p), pos_imp, neg_imp, self.skip_steps, self.hist_length)
            theta_ = np.mod(float(theta)+3.14, 6.28)
#             print('contin polar theta', theta_)
            oscillator_cpp.Phase_state(float(theta_), self.y, self.history, self.phase_oscillator, self.hist_length)
            oscillator_cpp.Make_biphasic_step(self.y, self.history, self.phase_oscillator, width_p, gap, round(kfactor*width_p), -pos_imp, neg_imp, self.skip_steps, self.hist_length)
        
        end_idx = oscillator_cpp.Return_end_idx(self.history, self.hist_length) - 1    
#         print('before end', end_idx)
        self.std = oscillator_cpp.Calc_std(self.history, start_idx, end_idx)

        self.current_step += 1
        self.done = self.current_step >= self.ep_length
        hists = self.get_history()
        self.reset2()
#         print('after reset end', oscillator_cpp.Return_end_idx(self.history, self.hist_length) )
        return theta, self.reward(pos_imp), self.done, hists
    
    def reset2(self):
        self.clear()
#         print('after clear end', oscillator_cpp.Return_end_idx(self.history, self.hist_length)  )
        oscillator_cpp.Make_step3(self.y, self.history, self.phase_oscillator, 1500*self.n_pulses, self.hist_length)  
        
    def clear(self):
        oscillator_cpp.Clear_state(self.y, self.nosc, self.epsilon, self.frrms)
        oscillator_cpp.Clear_history(self.history, self.hist_length)
        oscillator_cpp.Clear_phase_oscillator(self.phase_oscillator, self.hist_length)
    
    def reset(self):
        self.done = False
        self.current_step = 0 
        self.clear()
        theta = random.sample([2*np.pi/self.ep_length*i for i in range(self.ep_length)], k=1)
        oscillator_cpp.Make_step3(self.y, self.history, self.phase_oscillator, 1500*self.n_pulses, self.hist_length)
        if self.space_type == 'discrete':
            return np.array(theta)[0]
        return np.array(theta)

    def reward(self, amplitude):
        rews = self.std0 / self.std - self.ampl_reward * abs(amplitude)
        return rews