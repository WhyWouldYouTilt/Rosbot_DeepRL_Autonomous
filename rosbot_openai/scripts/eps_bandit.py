import numpy as np

class Tasks:
    def __init__(self,m):
        self.m = m
        self.mean = 0
        self.N = 0

    def choose(self):
        return np.random.randn()+self.m

    def update(self, x):
        self.N+=1
        print("N COUNT: %s", self.N)
        self.mean = (1-1.0/self.N)*self.mean+1.0/self.N*x

