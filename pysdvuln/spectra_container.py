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


class SpectraContainer():
    
    def __init__(self, spectra=[]):
        self.spectra = spectra
        self.ims = dict()

    
    def get_sas(self, period):
        if period == 0.:
            key = "PGA"
        else:
            key = "SA({:.1f})".format(period)
        if key in self.ims.keys():
            return self.ims[key]
        ims = list()
        for rec in self.spectra:
            ims.append(rec.get_sa(period))
        self.ims[key] = np.array(ims)
        return self.ims[key]
    
    
    @property
    def num_spectra(self):
        return len(self.spectra)
    
    
    def get_avgsas_T(self, struct_period):
        key = "avgSA({:.2f})".format(struct_period)
        if key in self.ims.keys():
            return self.ims[key]
        ims = list()
        for rec in self.spectra:
            ims.append(rec.get_avgsa_T(struct_period))
        self.ims[key] = np.array(ims)
        return self.ims[key]

    
    def get_avgsas(self, lower_period, upper_period):
        key = "avgSA({:.2f},{:.2f})".format(lower_period, upper_period)
        if key in self.ims.keys():
            return self.ims[key]
        ims = list()
        for rec in self.spectra:
            ims.append(rec.get_avgsa(lower_period, upper_period))
        self.ims[key] = np.array(ims)
        return self.ims[key]
    
    
    def plot_all_spectra(self, unit="g", x_scale="linear", allgrey=True):
        fig, ax = plt.subplots()
        ax.set_xlabel("Period (s)")
        if unit == "g":
            ax.set_ylabel("Acceleration (g)")
        elif unit == "m/s2":
            ax.set_ylabel("Acceleration (m/s2)")
        else:
            raise Exception("unit can only be 'g' or 'm/s2'")
        for r, rec in enumerate(self.spectra):
            spec = rec.get_spectrum()
            if unit == "g":
                spec[:,1] = spec[:,1]/9.81
            if allgrey:
                ax.plot(spec[:,0], spec[:,1], linewidth=0.5, color=[0.5,0.5,0.5])
            else:
                ax.plot(spec[:,0], spec[:,1], linewidth=0.5, label="id{}".format(r))
        ax.set_xscale(x_scale)
        if not allgrey:
            plt.legend()
        plt.show()
    

    def __repr__(self):
        return self.__str__()
        
    def __str__(self):
        return "<{} ".format(self.__class__.__name__) + \
               str(self.num_spectra) + " spectra>"
   
