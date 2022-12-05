import gym
from gym import spaces
import numpy as np
from ZernikeClass import ZernikeWF as zer


class SAO_Env(gym.Env):

    def __init__(self, Image, obs_range, action_range):

                              
        self.Image = Image
    
        
        #we create an observation space with predefined range
        self.observation_space = spaces.Box(low = obs_range[0], high = obs_range[1],
                                                                     dtype = np.float32)

        #similar to observation, we define action space 
        self.action_space = spaces.Box(low = action_range[0], shape=(5, ),high = action_range[1],
                                                                dtype = np.float32)
    def step(self, action): 
        
        Img = zer().generate_Img(self.Image, action) 
        
        reward = zer().metric_function(Img)
        state = [[reward]]                     #gives total sales based on spends
        done = True                            #Condition for completion of episode
        info = {}        

        return state, reward, done, info 

    def reset(self):
        obs = zer().metric_function(self.Image)

        return [[obs]]
    
