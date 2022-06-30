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

import numpy as np
from scipy.stats import norm
from scipy.stats import lognorm
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from pysdvuln.damg_dep_frag_models_noc import DamgDepFragModelsNoC
from pysdvuln.prob_collapse import ProbCollapse


def log(x, s, scale):
    return lognorm.cdf(x, s, 0., scale)


class DamgDepFragModels(DamgDepFragModelsNoC):
    
    def __init__(self, psdm, dsc, sigmab2b=0., imt="IM"):
        super().__init__(psdm, dsc, sigmab2b, imt)
        self.pc = ProbCollapse(psdm)


    def get_prob_coll(self, ims, thresh_ds1, unit="g"):
        # ProbCollapse works in g
        if unit == "g":
            ims_g = ims
        else:
            ims_g = ims/9.81
        # thresh = self.dsc.get_ds_thresh(ds1) # displacement-based threshold
        return self.pc(thresh_ds1*np.ones_like(ims_g), ims_g)
        

    def get_prob_no_coll(self, ims, thresh_ds1, unit="g"):
        return 1. - self.get_prob_coll(ims, thresh_ds1, unit)


    def get_fragility(self, ims, ds2, ds1=0, unit="m/s2"):
        '''
        this is tricky, everything else is done with IM in m/s2, while the
        alphas from Jalayer et al. (2017) use g.
        '''
        # previous version without uncertainty in ds threshold
        thresh_ds1 = self.dsc.get_ds_thresh(ds1) # displacement-based threshold
        if thresh_ds1 == 0.:
            thresh_ds1 = 1e-15
        mu, beta = self.get_frag_params(ds2, ds1, unit=unit, original=True)
        return norm.cdf((np.log(ims) - np.log(mu))/beta) * \
                self.get_prob_no_coll(ims, thresh_ds1, unit=unit) + \
                self.get_prob_coll(ims, thresh_ds1, unit=unit)
        # thresh_ds1, hist, bins = self.dsc.get_pmf_state(ds1)
        # thresh_ds1[thresh_ds1 == 0.] = 1e-15
        # thresh_ds2 = self.dsc.get_ds_thresh(ds2)
        # if thresh_ds2 == 0.:
        #     thresh_ds2 = 1e-15
        # mu, beta = self.get_frag_params_thresh(thresh_ds2, thresh_ds1, unit=unit)
        # if len(mu) == 1:
        #     return norm.cdf((np.log(ims) - np.log(mu))/beta) * \
        #            self.get_prob_no_coll(ims, thresh_ds1, unit=unit) + \
        #            self.get_prob_coll(ims, thresh_ds1, unit=unit)
        # else:
        #     out = list()
        #     for m, _ in enumerate(mu):
        #         out.append( (norm.cdf((np.log(ims) - np.log(mu[m]))/beta) * \
        #                      self.get_prob_no_coll(ims, thresh_ds1[m], unit=unit) + \
        #                      self.get_prob_coll(ims, thresh_ds1[m], unit=unit)) * hist[m])
        #     return np.sum(out, axis=0)


    def get_frag_params_thresh(self, thresh_ds2, thresh_ds1=0., unit="m/s2"):
        # this is to bypass damg_dep_frag_curves_noc that does not account for
        # aleatory uncertainty of ds_threhold
        # hysteretic energy given damage state thresholds
        E_ds1 = self.psdm.a * thresh_ds1 ** self.psdm.b
        E_ds2 = self.psdm.a * thresh_ds2 ** self.psdm.b
        # median values of fragility curves for DS-g2|DS-g1 (DS-g1==0, mainshock fragility)
        mu = ((E_ds2 - E_ds1) / \
             (self.psdm.c0*(1 - self.psdm.m * thresh_ds1))) ** (1/self.psdm.d)
        if unit == "g":
            return mu/9.81, self.beta
        return mu, self.beta


    def get_frag_params(self, ds2, ds1=0, unit="m/s2", original=False):
        '''
        ds1 = 0 returns the mainshock fragility
        ds1 != 0 returns the aftershock fragility for ds2
        '''
        if original:
            return super().get_frag_params(ds2, ds1, unit)
        if ds2 == 0:
            raise Exception("check ds2, it cannot be zero")
        if unit == "g":
            x_ims = self.x_ims_g
        else:
            x_ims = self.x_ims_ms2
        P_ds = self.get_fragility(x_ims, ds2, ds1, unit)
        popt, pcov = curve_fit(log, x_ims, P_ds, p0=[0.1, 1.])
        return popt[1], popt[0]


    def check_plots(self, unit="m/s2", imt=None, save=False, path=None):
        super().check_plots(unit, imt, save, path)
        
    
    def test(self, ds2, ds1=0, unit="m/s2"):
        if unit == "g":
            x_ims = self.x_ims_g
        else:
            x_ims = self.x_ims_ms2
        mu, beta = self.get_frag_params(ds2, ds1, unit=unit, original=True)
        thresh_ds1 = self.dsc.get_ds_thresh(ds1)
        thresh_ds2 = self.dsc.get_ds_thresh(ds2)
        mu2 = self.mu_ds(thresh_ds2, thresh_ds1)
        check = np.all(norm.cdf((np.log(x_ims) - np.log(mu))/beta) == \
                       norm.cdf((np.log(x_ims) - np.log(mu2))/self.beta))
        return check
    
    
    def __repr__(self):
        return self.__str__()
        
    def __str__(self):
        return "<{}>".format(self.__class__.__name__)
        
    
#%%

def get_inds(ds1):
    if ds1 == 0:
        inds = range(1,5)
        initial_ds = "Undamaged"
    elif ds1 == 1:
        inds = range(5,8)
        initial_ds = "DS1"
    elif ds1 == 2:
        inds = range(8,10)
        initial_ds = "DS2"
    elif ds1 == 3:
        inds = range(10,11)
        initial_ds = "DS3"
    return inds, initial_ds



class ComputedDamgDepFragModels(DamgDepFragModelsNoC):

    def __init__(self, frags_df, imt="IM", unit="g"):
        self.imt = imt
        imls = frags_df.iloc[:,0].to_numpy()
        if unit == "g":
            self.x_ims_g = imls
            self.x_ims_ms2 = imls*9.81
        elif unit == "m/s2":
            self.x_ims_g = imls/9.81
            self.x_ims_ms2 = imls
        self.frags = frags_df.iloc[:,1:].to_numpy()
        self.interpolants = list()
        for i in range(self.frags.shape[1]):
            self.interpolants.append(interp1d(self.x_ims_g*9.81,
                                              self.frags[:,i]))


    def get_fragility(self, ims, ds2, ds1=0, unit="m/s2"):
        #TODO this is preliminary
        if ds1 == 0:
            ind = ds2 - 1
        elif ds1 == 1:
            ind = 4 + ds2 - 2
        elif ds1 == 2:
            ind = 7 + ds2 - 3
        elif ds1 == 3:
            ind = 9
        if unit == "g":
            ims = ims*9.81
        return self.interpolants[ind](ims)


    def get_fragilities(self, ims=None, unit="m/s2"):
        ims = self.get_ims(ims, unit)
        P_ds = dict()
        for ds1 in range(4):
            for ds2 in range(ds1+1,5):
                P_ds[(ds1, ds2)] = self.get_fragility(ims, ds2, ds1, unit)
        return P_ds

