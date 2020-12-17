#coding:utf-8
import math
import numpy as np
from random import choice
from utility.tools import sigmoid
from math import log
from collections import defaultdict
from mf import MF
class BPR(MF):

    # BPR：Bayesian Personalized Ranking from Implicit Feedback
    # Steffen Rendle,Christoph Freudenthaler,Zeno Gantner and Lars Schmidt-Thieme

    def __init__(self):
        super(BPR, self).__init__()

    # def readConfiguration(self):
    #     super(BPR, self).readConfiguration()

    def initModel(self):
        super(BPR, self).initModel()
        self.config.lr = 0.01


    def train_model(self):

        print('Preparing item sets...')
        self.PositiveSet = defaultdict(dict)
        #self.NegativeSet = defaultdict(list)

        for user in self.rg.user:
            for item in self.rg.trainSet_u[user]:
                if self.rg.trainSet_u[user][item] >= 1:
                    self.PositiveSet[user][item] = 1
                # else:
                #     self.NegativeSet[user].append(item)
        print('training...')
        iteration = 0
        itemList = list(self.rg.item.keys())
        while iteration < self.config.maxIter:
            self.loss = 0
            for user in self.PositiveSet:
                u = self.rg.user[user]
                for item in self.PositiveSet[user]:
                    i = self.rg.item[item]
                    # if len(self.NegativeSet[user]) > 0:
                    #     item_j = choice(self.NegativeSet[user])
                    # else:
                    item_j = choice(itemList)
                    while (item_j in self.PositiveSet[user].keys()):
                        item_j = choice(itemList)
                    j = self.rg.item[item_j]
                    s = sigmoid(self.P[u].dot(self.Q[i]) - self.P[u].dot(self.Q[j]))
                    self.P[u] += self.config.lr * (1 - s) * (self.Q[i] - self.Q[j])
                    self.Q[i] += self.config.lr * (1 - s) * self.P[u]
                    self.Q[j] -= self.config.lr * (1 - s) * self.P[u]

                    self.P[u] -= self.config.lr * self.config.lambdaP * self.P[u]
                    self.Q[i] -= self.config.lr * self.config.lambdaQ * self.Q[i]
                    self.Q[j] -= self.config.lr * self.config.lambdaQ * self.Q[j]
                    self.loss += -log(s)
            self.loss += 0.5 * self.config.lambdaP * (self.P * self.P).sum() + 0.5 * self.config.lambdaQ * (self.Q * self.Q).sum()
            iteration += 1
            if self.isConverged(iteration):
                break

    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.rg.containsUser(u):
            u = self.rg.user[u]
            return self.Q.dot(self.P[u])
        else:
            return [self.rg.globalMean] * len(self.rg.item)

if __name__ == '__main__':
    gemf = BPR()
    gemf.init_model()
    gemf.train_model()
    gemf.predict_model()
