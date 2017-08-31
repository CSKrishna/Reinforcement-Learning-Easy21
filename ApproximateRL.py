
from Environment import *
import random 
import numpy as np



class ApproxRL(): #base class for initializing parameters necessary to learn using approximate state space
    def __init__(self, config, initializer, step, lamda, featurizer):
         self.e = config.e
         self.alpha = config.alpha
         self.m = len(config.actions)
         self.step = step #simulator for generating next state, reward for a given state-action pair
         self.init = initializer #initializes the state at the start of an episode
         self.featurizer = featurizer
         S = self.init()
         s = self.featurizer(S).shape
         self.W = np.array([np.zeros(s)]*self.m) # Q(s,a) = phi(s,a).T*W
         self.E = np.zeros_like(self.W) #eligibility trace
         self.lamda = lamda

    def max_action_value(self, s):  #returns argmax Q[(s,a)], max Q[(s,a)]
        q = np.sum(self.featurizer(s)*self.W, (1,2))
        a_max = np.argmax(q)
        v_max = np.max(q)
        return a_max, v_max
    
    def Qvalue(self, s, a): #returns Q[(s,a)]
        w = self.W[a, :, :]
        return np.sum(w*self.featurizer(s))
        
    
    def next_action(self, s): #returns action based on e-greedy policy
        probs = [self.e/self.m]*self.m
        a_max, _ = self.max_action_value(s)
        probs[a_max]  += 1 - self.e
        assert sum(probs) > 0.999, "probabilities don't sum to 1"
        return np.argmax(np.random.multinomial(1, probs))
    
    def getVtable(self): #returns 10 by 21 table of max values 
        V = np.zeros((10,21))
        for x in range(10):
            for y in range(21):
                s = (x+1, y+1)
                _, V[x,y] = self.max_action_value(s)
        return V

    def getQtable(self):
        Q = np.zeros((2,10,21))
        for x in range(10):
            for y in range(21):
                for a in range(2):
                    s = (x+1, y+1)
                    Q[a,x,y] = self.Qvalue(s,a)
        return Q
                
    
    def apply_control(self): #SARSA(Lambda) - Backward View for one episode
        S = self.init()
        A = self.next_action(S)
        i = 1 #switch 
        while S != "terminal":
            grad = self.featurizer(S)
            self.E[A, :,:] += grad
            S_, R = self.step(S, A)
            delta = R - self.Qvalue(S,A)
            if S_ != "terminal":         
                A_ = self.next_action(S_)
                delta += self.Qvalue(S_, A_)
                A = A_             
            else:
                i = 0  #to esnure E set to 0 before  start of new episode                     
            S = S_   
            self.W += (self.alpha*delta)*self.E
            self.E = (self.lamda*i)*self.E
            
    def Apply_SARSA(self, episodes = 1000): 
        for i in range(episodes):
            self.apply_control()
        self.trained = True


