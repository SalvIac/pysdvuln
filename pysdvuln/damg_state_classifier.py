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
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib import cm


class DamgStateClassifier():
    '''
    damage states from capacity according to Martins and Silva (2020)
    '''
    
    # Martins and Silva (2020) collapse is considered if the displacement
    # reaches 1.5 times the last displacement of the capacity curve
    COLLAPSE_FACTOR = 1.5    
    
    def __init__(self, thresholds=None, capacity_curve=None, cov=None,
                 **kwargs):
        self.__dict__.update(kwargs)
        if capacity_curve is not None and thresholds is not None:
            raise Exception("specify thresholds or capacity_curve")
        # thresholds from capacity curve or user-defined
        if capacity_curve is not None:
            self.mean_thresholds = self.get_displ_thresholds(capacity_curve)
        elif thresholds is not None:
            # if isinstance(thresholds, np.ndarray):
            #     thresholds = thresholds.tolist()
            self.mean_thresholds = np.array(thresholds)
            if not self.check_ascending_order(self.mean_thresholds):
                raise Exception("check threshold ascending order")
        else:
            raise Exception("cannot specify both thresholds and capacity_curve")
        # randomize thresholds according to lognormal distribution
        if cov is not None:
            self.cov = cov
            self.randomize = True
        else:
            self.randomize = False
            

    @classmethod
    def get_displ_thresholds(cls, capacity_curve):
        '''
        capacity curve defined as Sa[g]-Sd[m]
        displacement thresholds defined according to Martins and Silva (2020)
        '''
        if capacity_curve.shape[0] == 3:
            dy = capacity_curve[1,0]
        else:
            dy = capacity_curve[2,0]
        du = capacity_curve[-1,0]
        thresholds = [0.75*dy, 0.5*dy+0.33*du, 0.25*dy+0.67*du, du]
        return np.array(thresholds)


    @classmethod
    def check_ascending_order(cls, thresholds):
        if len(thresholds.shape) > 1:
            return np.all(np.sort(thresholds, axis=1) == thresholds, axis=1)
        else:
            return np.all(np.sort(thresholds) == thresholds)


    def get_damage_states(self):
        return np.arange(0, len(self.mean_thresholds)+1)
    
    
    def get_num_ds(self):
        return len(self.mean_thresholds)
    
    
    def get_damage_states_str(self):
        ds = self.get_damage_states()
        return ["DS{}".format(d) for d in ds]

    
    def get_ds_thresh(self, ds):
        if ds == 0:
            # DS=0 means undamaged structure
            return 0.
        return self.mean_thresholds[ds-1]
    

    def get_random_thresholds(self, size):
        mus = self.mean_thresholds
        sigmas = np.array(mus)*self.cov
        out = list()
        for mu, sigma in zip(mus, sigmas):
            out.append(np.exp(norm.rvs(loc=np.log(mu), scale=sigma, size=size)))
        out = np.array(out).T
        return out


    def get_thresholds(self, size, save=True):
        if not self.randomize:
            return self.mean_thresholds #+[1e9]
        else:
            np.random.seed(42) # for reproducibility
            out = self.get_random_thresholds(size)
            num = ~self.check_ascending_order(out)
            # check for ascending order
            while np.sum(num) != 0:
                out[num, :] = self.get_random_thresholds(np.sum(num))
                num = ~self.check_ascending_order(out)
            if save:
                if "rand_thresholds" in self.__dict__.keys():
                    warnings.warn("override rand_thresholds")
                self.rand_thresholds = out
            return out


    def get_pmf_state(self, ds, density=True):
        if ds < 1:
            return np.array([0.]), np.array([1.]), None
        if not self.randomize:
            xs = np.array([self.mean_thresholds[ds-1]])
            return xs, np.array([1.]), None
        else:
            if "thresholds_pmf" not in self.__dict__.keys():
                self.thresholds_pmf = self.get_thresholds(100000, save=False)
            hist, bins = np.histogram(self.thresholds_pmf[:,ds-1], bins=20)
            if density:
                hist = hist/100000
            xs = (bins[:-1] + bins[1:])/2
            return xs, hist, bins
        

    def classify(self, drift):
        if not self.randomize:
            return np.searchsorted(self.mean_thresholds, drift)
        else:
            out = list()
            thresholds = self.get_thresholds(len(drift))
            for d, drif in enumerate(drift):
                out.append( np.searchsorted(thresholds[d,:], drif) )
            return np.array(out)
    

    def check_collapse(self, drift):
        # return True if collapse
        if not isinstance(drift, np.ndarray):
            drift = np.array(drift)
        if not self.randomize:
            collapse_threshold = self.COLLAPSE_FACTOR*self.mean_thresholds[-1]
        else:
            if "rand_thresholds" not in self.__dict__.keys():
                _ = self.get_thresholds(len(drift)) # this creates rand_thresholds
            collapse_threshold = self.COLLAPSE_FACTOR*self.rand_thresholds[:,-1]
        return drift > collapse_threshold


    def get_ds_colors(self):
        ds_states = self.get_damage_states()
        m = cm.ScalarMappable(cmap=cm.jet)
        m.set_array(ds_states)
        ds_colors = m.to_rgba(ds_states) #["c", "g", "m", "b", "r"]
        return ds_states, ds_colors
    
    
    def check_plots(self):
        if not self.randomize:
            return
        
        thresholds = self.get_thresholds(100000, save=False)
        ds_states, ds_colors = self.get_ds_colors()
        
        # 3d plot of all distributions
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        for i in range(len(self.mean_thresholds)):
            hist, bins = np.histogram(thresholds[:,i], bins=20)
            xs = (bins[:-1] + bins[1:])/2
            ax.bar(xs, hist/100000, zs=i, zdir='y', alpha=0.8,
                   width=0.9*np.min(np.diff(bins)), color=ds_colors[i+1])
            ax.plot([self.mean_thresholds[i]]*2, [-0.1, i], [0.]*2, 
                    color=ds_colors[i+1])
        ax.set_yticks(range(len(self.mean_thresholds)))
        ax.set_yticklabels(self.get_damage_states_str()[1:])
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Damage State")
        ax.set_zlabel("Frequency")
        ax.set_ylim(-0.1, len(self.mean_thresholds)-1+0.1)
        ax.set_zlim(bottom=0.)

        # single plots
        for i in range(len(self.mean_thresholds)):
            # linear
            fig, ax = plt.subplots()
            # 2 lines equivalent to: xs, hist, bins = self.get_pmf_state(i+1, density=False)
            hist, bins = np.histogram(thresholds[:,i], bins=20)
            xs = (bins[:-1] + bins[1:])/2
            ax.bar(xs, hist, alpha=0.8, width=0.9*np.min(np.diff(bins)),
                   color=ds_colors[i+1])
            ax.set_xlabel("Threshold DS{}".format(i+1))
            ax.set_ylabel("Frequency")
            # log
            fig, ax = plt.subplots()
            hist, bins = np.histogram(np.log(thresholds[:,i]), bins=20)
            xs = (bins[:-1] + bins[1:])/2
            ax.bar(xs, hist, alpha=0.8, width=0.9*np.min(np.diff(bins)),
                    color=ds_colors[i+1])
            ax.set_xlabel("log(Threshold DS{})".format(i+1))
            ax.set_ylabel("Frequency")
        plt.show()


    def __repr__(self):
        return self.__str__()
        
    def __str__(self):
        return "<{}, thresholds: {}, randomize: {}>".format(self.__class__.__name__,
                                                            self.mean_thresholds.tolist(),
                                                            self.randomize)

    