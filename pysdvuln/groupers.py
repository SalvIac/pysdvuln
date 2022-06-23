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


class BaseGrouper():
    
    def __init__(self, centerbin, bins):
        '''
        centerbin: n periods for each category
        bins: numpy 1d array (n,)
        if value > bins[-1] then it returns the centerbin[-1]
        '''
        if len(centerbin) != len(bins):
            raise Exception("length of centerbin ({}) and bins ({}) are not consistent.".format(
                len(centerbin), len(bins)))
        self.centerbin = centerbin
        self.bins = bins
            
    def get_index(self, value):
        return np.searchsorted(self.bins, value)-1

    def __call__(self, value):
        return self.centerbin[self.get_index(value)]
    
    def __repr__(self):
        return self.__str__()
        
    def __str__(self):
        return "<{}, centerbin={}, bins={}>".format(
                self.__class__.__name__, self.centerbin, self.bins)
    


class PeriodGrouper(BaseGrouper):
    
    def __init__(self, periods=None, bins=None):
        # default for Iacoletti et al. (2022)
        if periods is None:
            periods = np.array([0.05, 0.2, 0.4, 0.6, 0.8, 1., 1.5, 2.])
            bins = np.hstack([[0.], periods[:-1] + np.diff(periods)/2])
            bins[1] = 0.1
        super().__init__(periods, bins)



class YieldCapGrouper(BaseGrouper):
    
    def __init__(self, yielding=None, bins=None):
        if yielding is None:
            # default impose a max of 3g
            yielding = np.arange(1., 3.1, 0.5) # in g
        if bins is None:
            bins = np.hstack([[0.], yielding[:-1]/1.])
        super().__init__(yielding, bins)


class YieldCapGrouper2(BaseGrouper):
    
    def __init__(self, yielding=None, bins=None):
        if yielding is None:
            # default impose a max of 3g
            yielding = np.arange(1., 3.1, 0.5) # in g
            bins = np.hstack([[0.], yielding[:-1]/2.])
        super().__init__(yielding, bins)


def duct2ry(ductility, period_y):
    corner_period2 = 1.
    if period_y < corner_period2:
        ry = np.sqrt(2*ductility-1)
    else:
        ry = ductility        
    return ry


