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
from scipy import interpolate
from scipy.stats import beta


class DamgDepVulnModels():

    def __init__(self, imt, ims, vulns, covs=None, unit="g"):
        self.imt = imt 
        if unit == "g": # in g by default
            self.ims = ims
        elif unit == "m/s2":
            self.ims = ims/9.81
        else:
            raise Exception("check units")
        self.vulns = vulns
        self.covs = covs
        
        
    @classmethod
    def from_ddfc_d2l(cls, ddfc, d2l):
        vulns = dict()
        covs = dict()
        for ds1 in range(0,4):
            frags = list()
            for ds2 in range(ds1+1, d2l.get_num_ds()+1): #TODO hardcoded 4 DSs
                frags.append( ddfc.get_fragility(ddfc.x_ims_g*9.81, ds2, ds1) )
            vulns[ds1] = cls.get_mlr(frags, d2l, ds1)
            var = cls.get_cov(frags, d2l, ds1)
            covs[ds1] = np.sqrt(var)/vulns[ds1]
        return cls(ddfc.imt, ddfc.x_ims_g, vulns, covs)


    @staticmethod
    def get_mlr(frags, d2l, ds1=0):
        if isinstance(frags, list):
            frags = np.array(frags).T
        temp1 = np.insert(frags, 0, np.ones_like(frags[:,0]), axis=1)
        temp2 = np.insert(frags, frags.shape[1], np.zeros_like(frags[:,0]), axis=1)
        prob_ds = temp1 - temp2
        mlr_mesh = np.meshgrid(d2l.get_mean_lr()[ds1:], np.ones_like(frags[:,0]))[0]
        mlr = np.sum( mlr_mesh * prob_ds, axis=1 )
        return mlr
    
    
    @staticmethod
    def get_cov(frags, d2l, ds1=0):
        # https://en.wikipedia.org/wiki/Law_of_total_variance
        if isinstance(frags, list):
            frags = np.array(frags).T
        temp1 = np.insert(frags, 0, np.ones_like(frags[:,0]), axis=1)
        temp2 = np.insert(frags, frags.shape[1], np.zeros_like(frags[:,0]), axis=1)
        prob_ds = temp1 - temp2
        mlr_mesh = np.meshgrid(d2l.get_mean_lr()[ds1:], np.ones_like(frags[:,0]))[0]
        vlr_mesh = np.meshgrid(d2l.get_var_lr()[ds1:], np.ones_like(frags[:,0]))[0]
        last = list()
        for k in range(mlr_mesh.shape[0]):
            vals = list()
            for i in range(1, mlr_mesh.shape[1]):
                for j in range(0, i):
                    vals.append( mlr_mesh[k,i] * prob_ds[k,i] * mlr_mesh[k,j] * prob_ds[k,j] )
            last.append( sum(vals) )
        var = np.sum( vlr_mesh * prob_ds, axis=1 ) + \
              np.sum( mlr_mesh**2 * (1-prob_ds) * prob_ds, axis=1 ) - \
              2*np.array(last)
        return var
    

    def get_ims(self, unit="g"):
        if unit == "g":
            ims = self.ims
        else:
            ims = self.ims*9.81
        return ims
    
    
    def get_vuln_curve_ds1(self, ds1):
        return self.vulns[ds1]
     
        
    def get_vuln_curves(self):
        return self.vulns
     

    def interpolate(self, iml, ds1=0):
        """
        interpolated mean loss ratios and covs
        """
        # gmvs are clipped to max(iml)
        gmvs_curve = np.piecewise(
                     iml, [iml > self.ims[-1]], [self.ims[-1], lambda x: x])
        ok = gmvs_curve >= self.ims[0]  # indices over the minimum #TODO
        curve_ok = gmvs_curve[ok]
        _means = np.zeros(gmvs_curve.shape[0])
        if "_mlr_i1d" not in self.__dict__.keys():
            self._mlr_i1d = dict()
        if ds1 not in self._mlr_i1d.keys():
            self._mlr_i1d[ds1] = interpolate.interp1d(self.ims, self.vulns[ds1])
        _means[ok] = self._mlr_i1d[ds1](curve_ok)
        return _means

     
    def get_vuln_curves_df(self, unit="g", uncertainty=False):
        ims = self.get_ims(unit)
        data = {self.imt: ims}
        for ds1 in self.vulns.keys():
            if ds1 == 0:
                label = "Und"
            else:
                label = "DS"+str(ds1)
            data["Mean|"+label] = self.vulns[ds1]
            if uncertainty and self.covs is not None:
                data["CoV|" +label] = self.covs[ds1]
        return pd.DataFrame(data)
        
    
    def plot(self, ax=None, unit="g", imt=None, max_img=None, dss=range(0,4),
             uncertainty=False, save=False, path=None, **kwargs):
        if max_img is None:
            inds = np.ones_like(self.ims, dtype=bool)
        else:
            inds = self.ims <= max_img
        if ax is None:
            fig, ax = plt.subplots(figsize=(6,6))
        else:
            save = False
        for ds1 in dss:
            # if ds1 == 0:
            #     label = "Undamaged state"
            # elif ds1 == 1:
            #     label = "Initial slight damage state"
            # elif ds1 == 2:
            #     label = "Initial moderate damage state"
            # elif ds1 == 3:
            #     label = "Initial extensive damage state" # "ds"+str(ds1)
            label = "DS"+str(ds1)
            color = ["g", "y", [1.,0.6,0.], "r"][ds1]
            if unit == "g":
                ims = self.ims
            else:
                ims = self.ims*9.81
            if uncertainty and self.covs is not None:
                mu = self.vulns[ds1]
                cov = self.covs[ds1]
                dist = BetaDistribution.get_distr(mu, cov*mu)
                ax.fill_between(ims[inds], dist.ppf(0.16)[inds], dist.ppf(0.84)[inds],
                                color=color, alpha=.1) # , label='16-84% confidence interval'
            ax.plot(ims[inds], self.vulns[ds1][inds], color=color, label=label,
                    **kwargs)
        ax.legend(framealpha=0.5)
        if imt is None:
            imt = self.imt
        ax.set_xlabel('{} ({})'.format(imt, unit))
        ax.set_ylabel('Mean Loss Ratio')
        if save:
            fig.savefig(os.path.join(path, "vuln.png"),
                        bbox_inches='tight', dpi=600, format="png")
        else:
            plt.show()


    def plot_covs(self, unit="g", imt=None, show=True, max_img=None):
        if max_img is None:
            inds = np.ones_like(self.ims, dtype=bool)
        else:
            inds = self.ims <= max_img
        fig = plt.figure()
        for ds1 in range(0,4):
            if ds1 == 0:
                label = "Undamaged state"
            elif ds1 == 1:
                label = "Initial slight damage state"
            elif ds1 == 2:
                label = "Initial moderate damage state"
            elif ds1 == 3:
                label = "Initial extensive damage state" # "ds"+str(ds1)

            if unit == "g":
                plt.plot(self.ims[inds], self.covs[ds1][inds], label=label)
            else:
                plt.plot(self.ims[inds]*9.81, self.covs[ds1][inds], label=label)
        plt.legend(framealpha=0.5)
        if imt is None:
            imt = self.imt
        plt.xlabel('{} ({})'.format(imt, unit))
        plt.ylabel('CoV Loss Ratio')
        if show:
            plt.show()

    
    def __repr__(self):
        return self.__str__()
        
    def __str__(self):
        return "<{}>".format(self.__class__.__name__)





class BetaDistribution():

    @classmethod
    def get_distr(cls, means, stddevs):
        a = cls._alpha(means, stddevs)
        b = cls._beta(means, stddevs)
        return beta(a, b)

    @staticmethod
    def _alpha(mean, stddev):
        return ((1 - mean) / stddev ** 2 - 1 / mean) * mean ** 2

    @staticmethod
    def _beta(mean, stddev):
        return ((1 - mean) / stddev ** 2 - 1 / mean) * (mean - mean ** 2)




