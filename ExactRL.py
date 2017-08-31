

from Environment import *
import random 
import numpy as np



class ExactRL(): #base class for initializing parameters necessary to learn using exact state space
    def __init__(self, config, initializer, step):
        self.config = config
        self.step = step #simulator for generating next state, reward for a given state-action pair
        self.init = initializer #initializes the state at the start of an episode
        self.Q = np.zeros((2, 10, 21)) #Q table
        self.N =  np.zeros((2, 10, 21)) #count of visits to each state, action pair 
        self.No = np.zeros((10,21)) #count of of visits to each state 
        self.e =  np.ones((10,21)) #probability of choosing non-greedy action, assymptotically converges to 0
        self.trained = False
        
    def max_action_value(self, s):  #returns argmax Q[(s,a)], max Q[(s,a)]
        s1 = s[0] - 1
        s2 = s[1] - 1
        a_max = np.argmax(self.Q[:, s1, s2])
        v_max = np.max(self.Q[:, s1, s2])
        return a_max, v_max
    
        
    def next_action(self, s): #returns action based on e-greedy policy
        s1 = s[0] - 1
        s2 = s[1] - 1
        m = len(self.config.actions)
        e = self.e[s1,s2]
        probs = [e/m]*m
        a_max, _ = self.max_action_value(s)
        probs[a_max]  += 1 - e
        assert sum(probs) > 0.999, "probabilities don't sum to 1"
        return np.argmax(np.random.multinomial(1, probs))
    
    def generate_episode(self): #generates episode for fixed e-greedy policy
        episode = []
        actions = self.config.actions
        s1 = self.init()
        while s1 != "terminal":
            a1 = self.next_action(s1)
            s2, r1 = self.step(s1, a1)
            episode.append((s1, a1, r1))
            s1 = s2         
        return episode
    
    def getVtable(self): #returns maximum values from given state
        return np.max(self.Q, axis = 0)




class MC_ExactRL(ExactRL): #Class for MC based RL
    
    def update_tables(self, episode):
        No = self.config.No
        G = episode[-1][2] #reward is same for all steps since there is no discounting
        for triplet in episode:
            s, a, _ = triplet
            s0 = s[0] - 1
            s1 = s[1] - 1
            self.N[a, s0, s1] += 1
            self.No[s0, s1] += 1
            self.Q[a, s0, s1] += 1./self.N[a, s0, s1]*(G -  self.Q[a, s0, s1])
            self.e[s0, s1] = No/(No + self.No[s0, s1])
            
    def Monte_Carlo_Control(self, episodes = 30000):
        for i in range(episodes):
            episode = self.generate_episode()
            #print (episode)
            self.update_tables(episode)
        self.trained = True        




class SarsaL_ExactRL(ExactRL): #Class for Sarsa(Lambda) - Backward View based RL
    
    def __init__(self, config, initializer, step, lamda):
        super().__init__(config, initializer, step)
        self.lamda = lamda
        self.E = np.zeros((2, 10, 21)) #Eligibility trace
        self.N[:,:,:] = 1 #to avoid division by 0
        
    def apply_control(self): #SARSA(Lambda) - Backward View for one episode
        S = self.init()
        A = self.next_action(S)
        No = self.config.No
        i = 1 #switch 
        while S != "terminal":
            s0 = S[0] - 1
            s1 = S[1] - 1         
            self.E[A, s0, s1] += 1
            self.N[A, s0, s1] += 1
            self.No[s0, s1] += 1
            self.e[s0, s1] = No/(No + self.No[s0, s1])
            S_, R = self.step(S, A)
            delta = R - self.Q[A, s0, s1]   
            if S_ != "terminal":
                s_0 = S_[0] - 1
                s_1 = S_[1] - 1                    
                A_ = self.next_action(S_)
                delta += self.Q[A_,s_0,s_1] 
                A = A_             
            else:
                i = 0                        
            S = S_   
            alpha = 1./self.N
            self.Q += alpha*delta*self.E
            self.E = self.lamda*self.E*i
            
    def Apply_SARSA(self, episodes = 1000): 
        for i in range(episodes):
            self.apply_control()
        self.trained = True

