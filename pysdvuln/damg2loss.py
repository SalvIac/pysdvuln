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

import warnings
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt


class Damg2Loss():
    
    damg = np.linspace(0,1,101)
    
    def __init__(self, mean_lr, cov_lr=None):
        self.mean_lr = np.array(mean_lr)
        if cov_lr is None:
            cov_lr = [0.]*len(mean_lr)
        self.cov_lr = np.array(cov_lr)
        if self.mean_lr[0] != 0.:
            warnings.warn("Adding 0 mean loss to represent DS0")
            self.mean_lr = np.insert(self.mean_lr, 0, 0.)
            self.cov_lr = np.insert(self.cov_lr, 0, 0.)
        
        
    @classmethod
    def default(cls):
        '''
        from Martins and Silva (2020): DS0, DS1, DS2, DS3, DS4
        '''
        return cls([0., 0.05, 0.2, 0.6, 1.], [0., 0.3, 0.2, 0.1, 0.])


    @classmethod
    def get_distr(cls, mu, cov):
        # https://stats.stackexchange.com/questions/12232/calculating-the-parameters-of-a-beta-distribution-using-the-mean-and-variance
        sigma = cov*mu
        a = ((1-mu)/sigma**2-1/mu) * mu**2
        b = a * (1/mu-1)
        return beta(a, b)


    def get_mean_lr(self):
        mean_lr = self.mean_lr
        # check with integrals
        # mean_lr = list()
        # for mu, cov in zip(self.mean_lr, self.cov_lr):
        #     lr, pr_lr = self.get_pmf(self.damg, mu, cov)
        #     mean_lr.append( np.sum(lr * pr_lr) )
        # using scipy.integrate
        # for mu, cov in zip(self.mean_lr, self.cov_lr):
        #     if mu == 0. or mu == 1. or cov == 0. or cov is None:
        #         mean_lr.append( mu )
        #     else:
        #         distr = self.get_distr(mu, cov)
        #         mean_lr.append( integrate.quad(lambda x: x*distr.pdf(x), 0, 1.)[0] )
        return mean_lr


    def get_cov_lr(self):
        return self.cov_lr


    def get_var_lr(self):
        var = list()
        for mu, cov in zip(self.mean_lr, self.cov_lr):
            if cov == 0. or cov is None:
                var.append(0.)
            else:
                distr = self.get_distr(mu, cov)
                _, v, _, _ = distr.stats(moments='mvsk')
                var.append(v)
        return np.array(var)
    

    @classmethod
    def get_pmf(cls, damg, mu, cov):
        if mu == 0. or mu == 1. or cov == 0. or cov is None:
            return np.array([mu]), np.array([1.])
        distr = cls.get_distr(mu, cov)
        return damg[:-1]+np.diff(damg)/2, np.diff(distr.cdf(damg))
        

    def plot(self):
        fig, ax = plt.subplots()
        for ds, (mu, cov) in enumerate(zip(self.mean_lr, self.cov_lr)):
            if mu != 0. and mu != 1.:
                distr = self.get_distr(mu, cov)
                ax.plot(self.damg, distr.pdf(self.damg)*0.01,
                        label="Distribution DS{}".format(ds))
            lr, pr_lr = self.get_pmf(self.damg, mu, cov)
            ax.bar(lr, pr_lr, width=0.009, alpha=0.7,
                   label="Distribution DS{}".format(ds))
        ax.set_xlabel("DS")
        ax.set_ylabel("Frequency")
        ax.legend()
        plt.show()
        
        
    def get_num_ds(self):
        return len(self.mean_lr)-1 # first is DS0 (we don't count it)
        
        
    def __repr__(self):
        return self.__str__()
        
    def __str__(self):
        return "<{}, mean: {}, cov: {}>".format(self.__class__.__name__,
                                                list(self.mean_lr),
                                                list(self.cov_lr))

