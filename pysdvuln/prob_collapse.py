# -*- coding: utf-8 -*-
# pysdvuln
# Copyright (C) 2021-2022 Salvatore Iacoletti
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
# 
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import curve_fit
from scipy.special import expit
from scipy.stats import logistic, binom
from sklearn.linear_model import LogisticRegression


class ProbCollapse():

    def __init__(self, psdm):
        self.psdm = psdm
        self.calculate()
        
        
    def calculate(self):
        # collapse cases
        self.coll_g1 = self.psdm.check_collapse(self.psdm.maxds_g1)
        # maxdisp vs collapses using the MS data only
        self.X1 = self.psdm.maxds_g1.reshape(1,-1).T
        self.y1 = np.zeros_like(self.coll_g1)
        self.y1[self.coll_g1] = 1.
        clf = LogisticRegression(C=1e5)
        clf.fit(np.log(self.X1), self.y1)
        self.a, self.b = clf.intercept_[0], clf.coef_[0][0]
        # if thresholds are not randomized, logistic function becomes 
        # Heaviside step function
        if not self.psdm.dam_state_cl.randomize:
            self.a *= 1e9
            self.b *= 1e9
        # im vs collapses using MS data only
        self.X2 = self.psdm.ims_g1.reshape(1,-1).T/9.81 # in g
        self.y2 = np.zeros_like(self.coll_g1)
        self.y2[self.coll_g1] = 1.
        clf = LogisticRegression(C=1e5)
        clf.fit(np.log(self.X2), self.y2)
        self.c0, self.d = clf.intercept_[0], clf.coef_[0][0]
        # # calibrate m using GM2 data        
        # m, pcov = curve_fit(lambda x, m: self.prob_coll(x, self.a, self.b, self.c0, self.d, m),
        #                          np.vstack([self.psdm.maxds_g1, self.psdm.ims_g2]).T, 
        #                          ~self.psdm.nocoll, p0=0.)
        #                     # bounds=(-np.inf, np.max(1/self.maxds_g1[filt_g2])))
        # if not self.psdm.dam_state_cl.randomize:
        # self.m = 1./(self.psdm.COLLAPSE_FACTOR*self.psdm.dam_state_cl.thresholds[-1])
        # else:
        self.m = 1./self.icdf_logistic(0.99, self.a, self.b)
        

    def __call__(self, gm1_max_drift, gm2_ims):
        return self.prob_coll(np.vstack([gm1_max_drift.flatten(), gm2_ims.flatten()]).T,
                              self.a, self.b, self.c0, self.d, self.m)
    

    # @classmethod
    # def prob_coll(cls, x, a, b, c0, d, m):
    #     drift1 = x[:,0]
    #     im2 = x[:,1]
    #     u = cls.eval_expit(drift1, a, b) # drift
    #     return np.clip(u + cls.eval_expit(im2, c0*(1-m*drift1), d), None, 1.)
    def prob_coll(cls, x, a, b, c0, d, m):
        drift1 = x[:,0]
        im2 = x[:,1]
        u = cls.eval_expit(drift1, a, b) # drift
        v = cls.eval_expit(im2, c0, d) # im
        temp = np.zeros_like(u)
        i0 = im2 == 0.
        d0 = drift1 == 0.
        noz = np.logical_and(drift1 != 0, im2 != 0)
        q = np.zeros_like(drift1[noz])
        q[ 1.-m*drift1[noz] >= 0. ] = np.sqrt(1.-m*drift1[ np.logical_and(noz, 
                                                     1.-m*drift1[noz] >= 0.) ])
        q[ 1.-m*drift1[noz] < 0. ] = 0. # to fix small numerical errors
        temp[noz] = u[noz] + (1.-u[noz]) / (1 + q * np.exp(-(c0 + d*np.log(im2[noz]))))
        temp[i0] = u[i0]
        temp[d0] = v[d0]
        return temp


    @classmethod
    def _negloglik(cls, x, y, a, b, c0, d, m):
        # https://stats.stackexchange.com/questions/266241/how-to-fit-a-generalized-logistic-function
        pi = cls.prob_coll(x, a, b, c0, d, m)
        # inds = np.logical_and(pi < 1, pi > 0)
        return -np.sum(binom.logpmf(y, 1, pi)) # -np.sum(np.log(pi**y * (1-pi)**(1-y)))
    
    
    def negloglik(self, m):
        return self._negloglik(np.vstack([self.psdm.maxds_g1, self.psdm.ims_g2]).T,
                                          ~self.psdm.nocoll,
                                          self.a, self.b, self.c0, self.d, m)
    

    @staticmethod
    def eval_expit(x, a, b):
        return expit(a + b*np.log(x))


    @staticmethod
    def cdf_logistic(x, a, b):
        # this is equivalent to eval_expit
        # https://stats.stackexchange.com/questions/403575/how-is-logistic-regression-related-to-logistic-distribution
        return logistic.cdf(np.log(x), loc=-a/b, scale=1/b)


    @staticmethod
    def icdf_logistic(y, a, b):
        # y between 0 and 1
        # logistic returns log(x), hence I return exp (unit acceleration: g)
        return np.exp(logistic.ppf(y, loc=-a/b, scale=1/b))


    def get_params(self, form="dict"):
        if form == "dict":
            return {"a": self.a,
                    "b": self.b,
                    "c0": self.c0,
                    "d": self.d,
                    "m": self.m}
        else:
            return self.a, self.b, self.c0, self.d, self.m
    

    def get_params_df(self):
        return pd.DataFrame(self.get_params("dict"), index=[0]).T


    def check_plots(self, unit="g", imt="IM", save=False, path=""):
        '''
        these plots are similar to the ones in Gentile and Galasso (2021)
        '''
        if unit == "g":
            scale = 1/9.81
        else:
            scale = 1.

        # 2d maxds-collapse
        fig, ax = plt.subplots()
        ax.scatter(self.X1, self.y1)
        Y = np.linspace(1e-3, np.max(self.psdm.maxds_g1), 500)
        ax.plot(Y, self.eval_expit(Y.flatten(), self.a, self.b), color="k")  
        ax.set_xlabel('Max drift G1')
        ax.set_ylabel('Probability of collapse')
        if save:
            fig.savefig(os.path.join(path, "pcoll_01.png"),
                        bbox_inches='tight', dpi=600, format="png")

        # 2d im-collapse
        fig, ax = plt.subplots()
        ax.scatter(self.X2*9.81*scale, self.y2)
        X = np.linspace(1e-3, np.max(self.psdm.ims_g2), 20)/9.81
        ax.plot(X*9.81*scale, self.eval_expit(X.flatten(), self.c0, self.d), color="k")
        ax.set_xlabel('{} G2 ({})'.format(imt, unit))
        ax.set_ylabel('Probability of collapse')
        if save:
            fig.savefig(os.path.join(path, "pcoll_02.png"),
                        bbox_inches='tight', dpi=600, format="png")
        
        # 3d maxds-im-collapse only GM1
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.scatter(np.zeros_like(self.X1[self.coll_g1]), self.X1[self.coll_g1], 1,
                   s=20, lw=0.5, color="r", label="Collapsed")
        ax.scatter(np.zeros_like(self.X1[~self.coll_g1]), self.X1[~self.coll_g1], 0.,
                   s=20, lw=0.5, color="b", label="Not collapsed")
        ax.scatter(self.X2[self.coll_g1]*9.81*scale, np.zeros_like(self.X2[self.coll_g1]), 1,
                   s=20, lw=0.5, color="r") #, label="Collapsed")
        ax.scatter(self.X2[~self.coll_g1]*9.81*scale, np.zeros_like(self.X2[~self.coll_g1]), 0.,
                   s=20, lw=0.5, color="b") #, label="Not collapsed")
    
        Y = np.linspace(1e-3, np.max(self.psdm.maxds_g1), 500)
        ax.plot(np.zeros_like(Y), Y, self.eval_expit(Y.flatten(), self.a, self.b),
                color="k")
        X = np.linspace(1e-3, np.max(self.psdm.ims_g2), 100)/9.81
        ax.plot(X*9.81*scale, np.zeros_like(X), self.eval_expit(X.flatten(), self.c0, self.d),
                color="k")
        ax.legend(framealpha=0.5)
        ax.set_xlabel('{} G2 ({})'.format(imt, unit))
        ax.set_ylabel('Max drift G1')
        ax.set_zlabel('Collapse/NoCollapse')
        ax.view_init(elev=25, azim=-140)
        if save:
            fig.savefig(os.path.join(path, "pcoll_03.png"),
                        bbox_inches='tight', dpi=600, format="png")

        
        # 3d maxds-im-collapse
        ds_states, ds_colors = self.psdm.get_ds_colors()
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.scatter(self.psdm.ims_g2[~self.psdm.nocoll]*scale,
                   self.psdm.maxds_g1[~self.psdm.nocoll], 1,
                   s=20, lw=0.5, color="r", label="Collapsed")
        ax.scatter(self.psdm.ims_g2[self.psdm.nocoll]*scale,
                   self.psdm.maxds_g1[self.psdm.nocoll], 0.,
                   s=20, lw=0.5, color="b", label="Not collapsed")
        for ds, col in zip(ds_states[1:], ds_colors[1:]):
            ys = self.psdm.dam_state_cl.get_ds_thresh(ds)*np.ones_like(X)
            z = self(ys, X)
            ax.plot(X*9.81*scale, ys, z, label="DS"+str(ds))    
        ax.plot(np.zeros_like(Y), Y, self.eval_expit(Y.flatten(), self.a, self.b),
                color="k")
        ax.plot(X*9.81*scale, np.zeros_like(X), self.eval_expit(X.flatten(), self.c0, self.d),
                color="k")
        XX = np.logspace(np.log10(1e-3), np.log10(np.max(self.psdm.ims_g2)), 200)/9.81
        YY = np.logspace(np.log10(1e-3), np.log10(np.max(self.psdm.maxds_g1)), 200)
        X2, Y2 = np.meshgrid(XX, YY)
        Z = self(Y2, X2)
        ax.plot_surface(X2*9.81*scale, Y2, Z.reshape(200,200), #cmap=cm.coolwarm,
                        linewidth=0, antialiased=False, alpha=0.3)
        ax.legend(framealpha=0.5)
        # ax.set_ylim([0., 0.5])
        ax.set_xlabel('{} G2 ({})'.format(imt, unit))
        ax.set_ylabel('Max drift G1')
        ax.set_zlabel('Collapse/NoCollapse')
        ax.set_zlim(0., 1.)
        ax.view_init(elev=25, azim=-140)
        if save:
            fig.savefig(os.path.join(path, "pcoll_04.png"),
                        bbox_inches='tight', dpi=600, format="png")


        # 2d im_g2-collapse
        fig, ax = plt.subplots()
        for ds, col in zip(ds_states[1:], ds_colors[1:]):
            ys = self.psdm.dam_state_cl.get_ds_thresh(ds)*np.ones_like(X)
            z = self(ys, X)
            ax.plot(X*9.81*scale, z, label="DS"+str(ds))
        ax.legend(framealpha=0.5)
        ax.set_xlabel('{} G2 ({})'.format(imt, unit))
        ax.set_ylabel('Probabilty of collapse')
        if save:
            fig.savefig(os.path.join(path, "pcoll_05.png"),
                        bbox_inches='tight', dpi=600, format="png")
        else:
            plt.show()
        
        
    def plot(self, unit="g", imt="IM", save=False, path=""):
        if unit == "g":
            scale = 1/9.81
        else:
            scale = 1.
        # 3d maxds-im-collapse
        ds_states, ds_colors = self.psdm.get_ds_colors()
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.scatter(self.psdm.ims_g2[~self.psdm.nocoll]*scale,
                   self.psdm.maxds_g1[~self.psdm.nocoll], 1,
                   s=20, lw=0.5, color="r", label="Collapsed")
        ax.scatter(self.psdm.ims_g2[self.psdm.nocoll]*scale,
                   self.psdm.maxds_g1[self.psdm.nocoll], 0.,
                   s=20, lw=0.5, color="b", label="Not collapsed")
        Y = np.linspace(1e-3, np.max(self.psdm.maxds_g1), 500)
        X = np.linspace(1e-3, np.max(self.psdm.ims_g2), 100)/9.81
        ax.plot(np.zeros_like(Y), Y, self.eval_expit(Y.flatten(), self.a, self.b),
                color="k")
        ax.plot(X*9.81*scale, np.zeros_like(X), self.eval_expit(X.flatten(), self.c0, self.d),
                color="k")
        XX = np.logspace(np.log10(1e-3), np.log10(np.max(self.psdm.ims_g2)), 200)/9.81
        YY = np.logspace(np.log10(1e-3), np.log10(np.max(self.psdm.maxds_g1)), 200)
        X2, Y2 = np.meshgrid(XX, YY)
        Z = self(Y2, X2)
        ax.plot_surface(X2*9.81*scale, Y2, Z.reshape(200,200), #cmap=cm.coolwarm,
                        linewidth=0, antialiased=False, alpha=0.3)
        # ax.set_ylim([0., 0.5])
        ax.set_xlabel('{} G2 ({})'.format(imt, unit))
        ax.set_ylabel('Max drift G1')
        ax.set_zlabel('Collapse/NoCollapse')
        ax.set_zlim(0., 1.)
        ax.view_init(elev=25, azim=-140)
        if save:
            fig.savefig(os.path.join(path, "pcoll.png"),
                        bbox_inches='tight', dpi=600, format="png") 
    
    
    def __repr__(self):
        return self.__str__()
        
    def __str__(self):
        return "<{}>".format(self.__class__.__name__)
    
    