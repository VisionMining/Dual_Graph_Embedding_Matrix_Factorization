# encoding:utf-8
import sys

sys.path.append("..")
import numpy as np
from mf import MF
from reader.trust import TrustGetter
from utility.matrix import SimMatrix
from utility.similarity import pearson_sp, cosine_sp
from utility import util


class SocialReg(MF):
    """
    docstring for SocialReg

    Ma H, Zhou D, Liu C, et al. Recommender systems with social regularization[C]//Proceedings of the fourth ACM international conference on Web search and data mining. ACM, 2011: 287-296.
    """

    def __init__(self):
        super(SocialReg, self).__init__()
        self.config.lr = 0.01
        temp = 0.7
        self.config.lambdaP = temp  # 0.03
        self.config.lambdaQ = temp  # 0.01
        self.config.lambdaB = temp  # 0.01
        self.config.alpha = 0.01
        self.config.isEarlyStopping = True
        self.tg = TrustGetter()
        self.init_model()

    def init_model(self):
        super(SocialReg, self).init_model()
        from collections import defaultdict
        self.user_sim_follower = SimMatrix()
        self.user_sim_followee = SimMatrix()
        print('constructing user-user similarity matrix...')

        for u in self.rg.user:
            for f in self.tg.get_followees(u):
                if self.user_sim_followee.contains(u, f):
                    continue
                s = self.get_sim(u, f)
                # sim = pearson_sp(self.rg.get_row(u), self.rg.get_row(f))
                # sim = round(sim, 5)
                self.user_sim_followee.set(u, f, s)

        for uu in self.rg.user:
            for ff in self.tg.get_followers(uu):
                if self.user_sim_follower.contains(uu, ff):
                    continue
                ss = self.get_sim(uu, ff)
                # sim = pearson_sp(self.rg.get_row(u), self.rg.get_row(f))
                # sim = round(sim, 5)
                self.user_sim_follower.set(uu, ff, ss)

    def get_sim(self, u, k):
        sim = (pearson_sp(self.rg.get_row(u), self.rg.get_row(k)) + 1.0) / 2.0  # fit the value into range [0.0,1.0]
        return sim

    def train_model(self):
        super(SocialReg, self).train_model()
        iteration = 0
        while iteration < self.config.maxIter:
            self.loss = 0
            for index, line in enumerate(self.rg.trainSet()):
                user, item, rating = line
                u = self.rg.user[user]
                i = self.rg.item[item]
                error = rating - self.predict(user, item)
                self.loss += 0.5 * error ** 2
                p, q = self.P[u], self.Q[i]
                bu, bi = self.Bu[u], self.Bi[i]

                social_term_p, social_term_loss = np.zeros((self.config.factor)), 0.0
                followees = self.tg.get_followees(user)
                for followee in followees:
                    if self.rg.containsUser(followee):
                        s = self.user_sim_followee[user][followee]
                        uf = self.P[self.rg.user[followee]]
                        social_term_p += s * (p - uf)
                        social_term_loss += s * ((p - uf).dot(p - uf))

                social_term_m = np.zeros((self.config.factor))
                followers = self.tg.get_followers(user)
                for follower in followers:
                    if self.rg.containsUser(follower):
                        s = self.user_sim_follower[user][follower]
                        ug = self.P[self.rg.user[follower]]
                        social_term_m += s * (p - ug)

                # update latent vectors
                self.Bu[u] += self.config.lr * (error - self.config.lambdaB * bu)
                self.Bi[i] += self.config.lr * (error - self.config.lambdaB * bi)
                self.P[u] += self.config.lr * (
                        error * q - self.config.alpha * (social_term_p + social_term_m) - self.config.lambdaP * p)
                self.Q[i] += self.config.lr * (error * p - self.config.lambdaQ * q)

                self.loss += 0.5 * self.config.alpha * social_term_loss

            self.loss += 0.5 * self.config.lambdaP * (self.P * self.P).sum() + 0.5 * self.config.lambdaQ * (
                    self.Q * self.Q).sum() + 0.5 * self.config.lambdaB * ((self.Bu * self.Bu).sum() + (self.Bi * self.Bi).sum())

            iteration += 1
            if self.isConverged(iteration):
                break


if __name__ == '__main__':
    tcsr = SocialReg()
    tcsr.train_model()
    rmse, mae = tcsr.predict_model()
