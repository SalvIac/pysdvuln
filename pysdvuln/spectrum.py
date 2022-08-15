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

from copy import deepcopy
import numpy as np
from scipy import interpolate
from scipy.stats import gmean
import matplotlib.pyplot as plt


class Spectrum():
    '''
    Response spectrum
    '''
    
    def __init__(self, periods, psa, unit="m/s2", correct_pga=False, **kwargs):
        '''
        always store response spectrum in m/s2
        '''
        if unit not in ["g", "m/s2"]:
            raise Exception("unit can only be 'g' or 'm/s2'")
        self.periods = np.array(periods)
        if unit == "g":
            self.psa = np.array(psa)*9.81
        else:
            self.psa = np.array(psa)
        if correct_pga:
            self.correct_pga()
        self.kwargs = kwargs
        self.__dict__.update(kwargs)


    def correct_pga(self):
        if self.periods[0] != 0:
            self.periods = np.insert(self.periods, 0, 0.)
            self.psa = np.insert(self.psa, 0, self.psa[0])
        

    def scale(self, scaling_factor):
        psa = np.abs(scaling_factor) * self.psa
        kwargs = deepcopy(self.kwargs)
        return self.__class__(self.periods, psa, unit="m/s2", **kwargs)

    
    @property
    def spectrum(self):
        return np.array([self.periods, self.psa]).T # format 2d array

    
    def get_spectrum(self, unit="m/s2"):
        if unit == "g":
            spe = self.spectrum
            spe[:,1] = spe[:,1]/9.81
            return spe
        elif unit == "m/s2":
            return self.spectrum
        else:
            raise Exception("unit can only be 'g' or 'm/s2'")


    def get_sa(self, period):
        if "interp_spectra" not in self.__dict__.keys():
            self.interp_spectra = self.get_interp_spectra()
        if isinstance(period, float):
            return float(self.interp_spectra(period))
        else:
            return self.interp_spectra(period)


    def get_avgsa_T(self, struct_period):
        '''
        The avgSa, selected here as IM, is conventionally calculated by
        considering a range of 10 equally-spaced periods spanning 
        approximately from a lower bound of 0.2T1 and an upper bound of 1.5T1,
        where T1 is the fundamental period of the structure (e.g., Kohrangi et 
        al. 2016).
        '''
        lower_period = 0.2*struct_period
        upper_period = 1.5*struct_period
        return self.get_avgsa(lower_period, upper_period)


    def get_avgsa(self, lower_period, upper_period):
        periods = np.linspace(lower_period, upper_period, 10)
        if "interp_spectra" not in self.__dict__.keys():
            self.interp_spectra = self.get_interp_spectra()
        return gmean( self.interp_spectra(periods) )


    def get_interp_spectra(self):
        '''
        spectra is a 2d nump array (n,2), with n number of response periods
        '''
        spectrum = self.get_spectrum()
        f = interpolate.interp1d(spectrum[:,0], spectrum[:,1])
        return f


    def plot_response_spectrum(self, ax=None):
        period_sa = self.get_spectrum()
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(period_sa[:,0], period_sa[:,1])
        ax.set_xlabel("Response period (s)")
        ax.set_ylabel("Acceleration (m/s2)")
        if ax is None:
            plt.show()
        return ax


    def __repr__(self):
        return self.__str__()
        
    def __str__(self):
        return "<{}".format(self.__class__.__name__) +">"

