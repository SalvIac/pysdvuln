# -*- coding: utf-8 -*-
"""
@author: Salvatore Iacoletti
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from scipy.special import expit
from pysdvuln.damg_dep_frag_curves_noc import DamgDepFragCurvesNoC


class DamgDepFragCurves(DamgDepFragCurvesNoC):
    
    def __init__(self, psdm, dsc, sigmab2b=0.):
        super().__init__(psdm, dsc, sigmab2b)
        self.alphas = dict()
        
        # ds1=0, i.e., mainshocks
        ds4 = psdm.ds_g1 == 4
        X = np.log(psdm.ims_g1.reshape(1,-1).T/9.81) # in g
        y = np.zeros_like(X[:,0])
        y[ds4] = 1.
        clf = LogisticRegression(C=1e5)
        clf.fit(X, y)
        self.alphas[0] = (clf.intercept_[0], clf.coef_[0][0])
        
        # ds1>0 
        for ds1 in range(1,4):
            bla = self.psdm.ds_g1 == ds1
            ds4 = self.psdm.ds_g2[bla] == 4
            X = np.log(self.psdm.ims_g2[bla].reshape(1,-1).T/9.81) # in g
            y = np.zeros_like(X[:,0])
            y[ds4] = 1.
            clf = LogisticRegression(C=1e5)
            clf.fit(X, y)
            self.alphas[ds1] = (clf.intercept_[0], clf.coef_[0][0])
    
    
    def get_alphas(self, ds1):
        if ds1 == 4:
            raise Exception("cannot get alpha for ds1=4")
        return self.alphas[ds1]


    def get_prob_coll(self, ims, ds1=0, unit="g"):
        if unit == "g":
            ims_g = ims
        else:
            ims_g = ims/9.81
        alphas = self.get_alphas(ds1)
        return expit(alphas[0] + np.log(ims_g) * alphas[1])
        

    def get_prob_no_coll(self, ims, ds1=0, unit="g"):
        return 1. - self.get_prob_coll(ims, ds1, unit)


    def get_fragility(self, ims, ds2, ds1=0, unit="m/s2"):
        '''
        this is tricky, everything else is done with IM in m/s2, while the
        alphas from Jalayer et al. (2017) use g.
        '''
        if unit == "g":
            ims = ims*9.81 # convert to m/s2
        mu, beta = self.get_frag_params(ds2, ds1)
        return norm.cdf((np.log(ims) - np.log(mu))/beta) * \
               self.get_prob_no_coll(ims, ds1, unit="m/s2") + \
               self.get_prob_coll(ims, ds1, unit="m/s2")



    def check_plots(self, unit="m/s2", imt="IM"):
        super().check_plots(unit)

        X_test = np.linspace(0.01, 15., 100) # in g
        for ds1 in range(0,4):
            alphas = self.get_alphas(ds1)
            if ds1 == 0:
                ds4 = self.psdm.ds_g1 == 4
                X = self.psdm.ims_g1.reshape(1,-1).T/9.81 # in g
                y = np.zeros_like(X[:,0])
                y[ds4] = 1.
            else:
                bla = self.psdm.ds_g1 == ds1
                ds4 = self.psdm.ds_g2[bla] == 4
                X = self.psdm.ims_g2[bla].reshape(1,-1).T/9.81 # in g
                y = np.zeros_like(X[:,0])
                y[ds4] = 1.
            plt.figure()
            plt.scatter(X.ravel(), y, s=5, color="k", zorder=20)
            plt.plot(X_test, self.get_prob_coll(X_test, ds1=ds1, unit="g"), 
                     color="r", lw=1)
            plt.plot(X_test, self.get_prob_no_coll(X_test, ds1=ds1, unit="g"), 
                     color="r", lw=1, ls="--")
            plt.ylabel("Collpase/NoCollapse")
            plt.xlabel("{} (g)".format(imt))
            plt.show()
            print("ds{}".format(ds1), np.sum(y), y.shape[0],
                  np.sum(y)/y.shape[0], alphas)
    
    def __repr__(self):
        return self.__str__()
        
    def __str__(self):
        return "<{}>".format(self.__class__.__name__)
        

