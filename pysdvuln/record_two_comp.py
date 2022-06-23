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
import matplotlib.pyplot as plt
from pysdvuln.record import Record


class RecordChooseComp(Record):
    '''
    Ground-motion record container for two horizontal components
    '''
    
    def __init__(self, rec1, rec2):
        '''
        gmr: Record
        kwargs: other information if needed
        '''
        if np.argmax([rec1.pga, rec2.pga]) == 0:
            super().__init__(rec1.gmr, rec1.t_step, **rec1.kwargs)
        else:
            super().__init__(rec2.gmr, rec2.t_step, **rec2.kwargs)
        self.complete = RecordTwoComp(rec1, rec2)



class RecordTwoComp(Record):
    '''
    Ground-motion record container for two horizontal components
    '''
    
    def __init__(self, rec1, rec2, **kwargs):
        '''
        gmr: Record
        kwargs: other information if needed
        '''
        self.rec1 = rec1
        self.rec2 = rec2
        self.pga = self.combine(rec1.pga, rec1.pga, mode="geomean")
        self.pgas = [rec1.pga, rec1.pga]
        self.kwargs = kwargs
        self.__dict__.update(kwargs)
    
    
    def get_spectrum(self, periods=None, mode="geomean"):
        temp1 = self.rec1.generate_response_spectrum(periods)
        out = self.combine(temp1[:,1],
                           self.rec2.generate_response_spectrum(periods)[:,1],
                           mode)
        return np.vstack([temp1[:,0], out]).T


    @classmethod
    def combine(cls, val1, val2, mode="geomean"):
        if mode == "geomean":
            out = np.sqrt(val1*val2)
        elif mode == "srss":
            out = np.sqrt(val1**2 + val2**2)
        else:
            raise Exception("check mode")
        return out
    
    
    def choose(self, val1, val2):
        if np.argmax([val1, val2]) == 0:
            return Record(self.rec1.gmr, self.rec1.t_step, **self.rec1.kwargs)
        else:
            return Record(self.rec2.gmr, self.rec2.t_step, **self.rec2.kwargs)
        
    
    def plot_inputs(self, unit="m/s2"):
        fig, axs = plt.subplots(2,1)
        self.rec1.plot_inputs(unit, axs[0])
        self.rec2.plot_inputs(unit, axs[1])
        plt.show()
        return axs
    
    
    def plot_response_spectrum(self, periods=None, ax=None):
        ax = super().plot_response_spectrum(periods, ax)
        period_sa = self.rec1.get_spectrum(periods)
        ax.plot(period_sa[:,0], period_sa[:,1])
        period_sa = self.rec2.get_spectrum(periods)
        ax.plot(period_sa[:,0], period_sa[:,1])
        return ax
    
    
    def scale(self, scaling_factor):
        rec1 = self.rec1.scale(scaling_factor)
        rec2 = self.rec2.scale(scaling_factor)
        return self.__class__(rec1, rec2, **self.kwargs)
    
    
    def __str__(self):
        string = "<{} ".format(self.__class__.__name__) + \
                 str(self.rec1) + ", " + str(self.rec2)
        return string+">"
    
    
    @property
    def num_points(self):
        raise NotImplementedError

    @property
    def duration(self):
        raise NotImplementedError

    def get_significant_duration(self, start=0.05, end=0.95, se=False):
        raise NotImplementedError

    @property
    def time(self):
        raise NotImplementedError

    def get_eqsig_record(self):
        raise NotImplementedError

    def generate_acc_vel_disp_series(self):
        raise NotImplementedError

    def plot_comparison_spectra(self):
        raise NotImplementedError

    def plot_acc_vel_disp(self, unit="m/s2", axs=None):
        raise NotImplementedError

