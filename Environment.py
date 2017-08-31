

import random 
import numpy as np



class config(object): #hyper-parameters for planning, control in the Easy21 MDP
    e = 0.05
    alpha = 0.01
    actions = {'hit': 0, 'stick': 1}     
    No = 100


def coarse_featurizer(s):
    assert (s[0] < 11) & (s[0] > 0), "state not within range"
    assert (s[1] < 22) & (s[1] > 0), "state not within range"
    
    n1 =  3
    n2 = 6
    features = np.zeros((n1,n2), dtype=np.int16)    
    s1 = []
    if (s[0] <= 4)  & (s[0] >= 1):
        s1.append(0)
    if (s[0] >= 4)  & (s[0] <= 7):
        s1.append(1)
    if (s[0] >= 7) & (s[0] <= 10):
        s1.append(2)
    
    s2 = []
    if (s[1] >= 1) & (s[1] <= 6):
        s2.append(0)
    if (s[1] >= 4) & (s[1] <= 9):
        s2.append(1)
    if (s[1] >= 7) & (s[1] <= 12):
        s2.append(2)
    if (s[1] >= 10) & (s[1] <= 15):
        s2.append(3)
    if (s[1] >= 13) & (s[1] <= 18):
        s2.append(4)
    if (s[1] >= 16) & (s[1] <= 21):
        s2.append(5)
        
    features[s1,s2] = 1
    return features
        
        



def exact_featurizer(s):
    assert (s[0] < 11) & (s[0] > 0), "state not within range"
    assert (s[1] < 22) & (s[1] > 0), "state not within range"
    
    n1 =  10
    n2 = 21
    features = np.zeros((n1,n2), dtype=np.int16)       
    s1 = s[0] -1
    s2 = s[1] - 1
    features[s1,s2] = 1
    return features
    


class simulator(): #defines dynamics of MDP: generating next state, reward for given state-action; intializing an episode
    def __init__(self, weights = [1,2]):
        self.weights = weights
    
    def draw(self):
        return  random.randint(1,10)*random.choices([-1, 1], self.weights)[0]  
    
    def init(self): #initialize start start of new episode
         return (random.randint(1,10), random.randint(1,10))
    
    def step(self, state, action):#generate next step of an episode
        if action == 1 :
            deal = state[0]
            while ((deal < 17) & (deal >= 1)):
                deal += self.draw()
            if (deal > 21 or deal < 1):
                return "terminal", 1
            if deal > state[1]:
                return "terminal", -1
            if deal == state[1]:
                return "terminal", 0 
            if deal < state[1]:
                return "terminal", 1 
        else:
            deal = state[1] + self.draw()
            if (deal > 21 or deal < 1): 
                return "terminal", -1
            else: 
                return (state[0], deal), 0 
            


