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


class RecordPair():
    '''
    Container for a pair of ground-motion records
    '''
    
    LAG_T_STEP = 0.1
    
    def __init__(self, rec1, rec2, sf1=1., sf2=1., lag=40., rest=10.):
        '''
        rec1: ground motion record 1 (first) - instance of Record
        rec2: ground motion record 2 (second) - instance of Record
        lag: time (s) between rec1 and rec2
        '''
        self.rec1_orig = rec1
        self.rec2_orig = rec2
        self.sf1 = sf1
        self.sf2 = sf2
        if sf1 == 1.:
            self.rec1 = rec1
        else:
            self.rec1 = rec1.scale(sf1)
        if sf2 == 1.:
            self.rec2 = rec2
        else:
            self.rec2 = rec2.scale(sf2)
        self.lag = lag
        self.rest = rest
    
    @property
    def time_lag(self):
        time1 = self.rec1.time
        time_lag = time1[-1] + self.time_arange(self.lag)
        return time_lag
    

    def time_arange(self, time):
        out = np.arange(self.LAG_T_STEP, time + self.LAG_T_STEP, self.LAG_T_STEP)
        return out

    
    @property
    def end_rest_time(self):
        return self.time_lag[-1]
    
    @property
    def time(self):
        time1 = self.rec1.time
        time_lag = time1[-1] + self.time_arange(self.lag)
        time2 = time_lag[-1] + self.LAG_T_STEP + self.rec2.time
        time_rest = time2[-1] + self.time_arange(self.rest)
        return np.hstack([time1, time_lag, time2, time_rest])
    
    @property
    def num_points(self):
        return len(self.time)

    @property
    def duration(self):
        return self.time[-1]

    @property
    def gmr(self):
        return np.hstack([self.rec1.gmr, [0.]*self.time_arange(self.lag).shape[0],
                          self.rec2.gmr, [0.]*self.time_arange(self.rest).shape[0]])


    def get_gmr(self, unit="m/s2"):
        if unit == "g":
            return self.gmr/9.81
        elif unit == "m/s2":
            return self.gmr
        else:
            raise Exception("unit can only be 'g' or 'm/s2'")


    def get_time_gmr(self, unit="m/s2"):
        return np.vstack([self.time, self.get_gmr(unit)]).T


    def plot_acc(self, unit="m/s2", ax=None):
        '''
        quick and dirty plot to check
        '''
        if ax is None:
            fig, ax = plt.subplots()
        ax.set_xlabel("Time (s)")
        if unit == "g":
            ax.set_ylabel("Acceleration (g)")
        elif unit == "m/s2":
            ax.set_ylabel("Acceleration (m/s2)")
        else:
            raise Exception("unit can only be 'g' or 'm/s2'")
        ax.plot(self.time, self.get_gmr(unit))
        plt.show()
        return ax


    def __repr__(self):
        return self.__str__()
        
    def __str__(self):
        string = "<{} ".format(self.__class__.__name__) + \
               str(self.rec1) + ", " + \
               "lag: {:.0f}".format(self.lag) + "s, " + \
               str(self.rec2)
        return string+">"

