"""
Template for implementing QLearner  (c) 2015 Tucker Balch
"""

import numpy as np
import random as rand

class QLearner(object):

    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False):

        self.verbose = verbose
        self.num_actions = num_actions
        self.s = 0
        self.a = 0
        self.num_states = num_states
        self.alpha = alpha
        self.gamma =gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.q = np.array([[0]*self.num_actions]*self.num_states , dtype=float)
        self.T , self.visited_states , self.T_c , self.R = None,None,None,None
        if dyna >0:
            self.T = np.array([[[0]*self.num_states]*self.num_actions]*self.num_states , dtype=float)
            self.visited_states = set([])
            self.T_c = np.array([[[0]*self.num_states]*self.num_actions]*self.num_states , dtype=float)
            self.R = np.array([[0]*self.num_actions]*self.num_states , dtype=float)

    def author(self):
        return 'snagamalla3' # replace tb34 with your Georgia Tech username.


    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        action = None
        if rand.uniform(0,1) > self.rar:
            action = np.argmax(self.q[s])
        else:
            action = rand.randint(0, self.num_actions-1)
        if self.verbose: print "s =", s,"a =",action
        self.s  = s
        self.a =action
        if self.dyna >0:
            self.visited_states.add((s,action))
        return action


    def dynaLearn(self):
        sampleStatesInd = np.random.choice(range(len(self.visited_states)),self.dyna)
        visited_states_list = list(self.visited_states)
        dynaCount = self.dyna
        samplingCount = max(1,int(0.5*self.num_states))
        for i in range(dynaCount):
            s,a = visited_states_list[sampleStatesInd[i]]
            T_s_a = self.T[s][a]
            s_prime =np.argmax(np.random.multinomial(samplingCount,T_s_a.tolist(), size =1) )
            r = self.R[s][a]
            self.q[s][a] = (1 - self.alpha)*self.q[s][a] + self.alpha*(r + self.gamma*self.q[s_prime][np.argmax(self.q[s_prime])])




    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """
        self.q[self.s][self.a] = (1 - self.alpha)*self.q[self.s][self.a] + self.alpha*(r + self.gamma*self.q[s_prime][np.argmax(self.q[s_prime])])

        if self.dyna > 0:
            self.R[self.s][self.a] =  (1 - self.alpha)*self.R[self.s][self.a] + self.alpha*r
            self.T_c[self.s][self.a][s_prime] += 1
            self.T[self.s][self.a] = self.T_c[self.s][self.a]/np.sum(self.T_c[self.s][self.a])
            self.visited_states.add((self.s,self.a))

        action = None
        if rand.uniform(0,1) > self.rar:
            action = np.argmax(self.q[s_prime])
        else:
            action = rand.randint(0, self.num_actions-1)

        self.s =s_prime
        self.a = action
        self.rar =self.radr*self.rar
        if self.dyna >0:
            self.dynaLearn()
        if self.verbose: print "s =", s_prime,"a =",action,"r =",r
        return action

if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
