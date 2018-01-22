# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 14:39:44 2017

@author: rbarnes

This class implements the breach radar algorithm for detecting unknown common points of compromise.
"""
import numpy as np


class BreachRadar:
    def __init__(self, N, epsilon=0.001, alpha=5, beta=5):
        self.alpha = alpha
        self.beta = beta
        self.N = N
        self.epsilon = epsilon
        self.UniformBlames()
        self.theta = np.array([1/self.N.shape[1]]*self.N.shape[1])

    def FindBreaches(self,frauds):
        iterations = 0
        previous_theta = np.ones(self.theta.shape)
        while np.abs(previous_theta-self.theta).sum() >= self.epsilon:
            iterations += 1
            self.UpdateBlames(frauds)
            previous_theta = np.array([obj for obj in self.theta])
            self.UpdatePOCProbabilities()
            print("Iteration: {0} Score: {1}".format(iterations,np.abs(previous_theta-self.theta).sum()))
        return(self.theta, self.B)

    def UpdateBlames(self,frauds):
        for i in range(self.N.shape[0]):
            denom = 1
            for k in range(self.N.shape[1]):
                denom += self.theta[k]*self.N[i, k]
            for j in range(self.N.shape[1]):
                self.B[i, j] = frauds[i,0]*self.N[i,j]*self.theta[j]/(denom)

    def UpdatePOCProbabilities(self):
        for j in range(self.N.shape[1]):
            z = self.B[:,j].sum()
            self.theta[j] = (z + self.alpha)/(self.N.shape[0] +
                                              self.alpha +
                                              self.beta)

            #self.theta = (self.theta/self.theta.sum())

    def UniformBlames(self):
        j = self.N.shape[1]
        self.B = np.matrix([[1/j]*j for i in range(self.N.shape[0])])


if __name__ == '__main__':
    import time
    import pandas as pd
    t0 = time.time()
    df = pd.read_csv('h:/desktop/detail analysis.txt', sep='~')
    t1 = time.time()
    G = np.matrix(pd.pivot_table(
                                    df,
                                    index=['Card Number'],
                                    columns=['Merchant Corporate Name'],
                                    aggfunc=len
                                ).fillna(0)['Metrics'])
    t2 =time.time()
    model = BreachRadar(G, 0.0001, 0.5, 0.5)
    x = G[:, 121]
    t3 = time.time()
    model.FindBreaches(x)
    t4=time.time()
    print(np.argmax(model.theta))
    print(model.theta[np.argmax(model.theta)]/np.mean(model.theta))
    print("Loading Dataset:\t\t{0}s".format(t1-t0))
    print("Pivoting Dataset:\t\t{0}s".format(t2-t1))
    print("Setting Up Algo:\t\t{0}s".format(t3-t2))
    print("Runnning Algo:\t\t{0}s".format(t4-t3))
    print("Total Wall Time:\t\t{0}s".format(t4-t0))