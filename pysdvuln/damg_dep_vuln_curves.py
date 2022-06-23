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


class DamgDepVulnCurves():
    
    
    def __init__(self, ddfc, d2l):
        self.imt = ddfc.imt
        self.ims = ddfc.x_ims_g # in g by default
        self.vulns = dict()
        for ds1 in range(0,4):
            frags = list()
            for ds2 in range(ds1+1, 5):
                frags.append( ddfc.get_fragility(self.ims*9.81, ds2, ds1) )
            frags = np.array(frags).T
            temp1 = np.insert(frags, 0, np.ones_like(frags[:,0]), axis=1)
            temp2 = np.insert(frags, frags.shape[1], np.zeros_like(frags[:,0]), axis=1)
            prob_ds = temp1 - temp2
            dmg2loss_mesh = np.meshgrid(d2l.get_mean_loss()[ds1:], np.ones_like(frags[:,0]))[0]
            self.vulns[ds1] = np.sum( prob_ds * dmg2loss_mesh, axis=1 )
    

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
     
     
    def get_vuln_curves_df(self, unit="g"):
         ims = self.get_ims(unit)
         data = {self.imt: ims}
         for ds1 in self.vulns.keys():
             if ds1 == 0:
                 label = "Mainshock"
             else:
                 label = "Initial DS"+str(ds1)
             data[label] = self.vulns[ds1]
         return pd.DataFrame(data)
        
    
    def check_plots(self, unit="g", imt=None, save=False, path=None,
                    max_img=None):
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
                plt.plot(self.ims[inds], self.vulns[ds1][inds], label=label)
            else:
                plt.plot(self.ims[inds]*9.81, self.vulns[ds1][inds], label=label)
        plt.legend(framealpha=0.5)
        if imt is None:
            imt = self.imt
        plt.xlabel('{} ({})'.format(imt, unit))
        plt.ylabel('Mean Loss Ratio')
        if save:
            fig.savefig(os.path.join(path, "vuln.png"),
                        bbox_inches='tight', dpi=600, format="png")
        else:
            plt.show()

    
    def __repr__(self):
        return self.__str__()
        
    def __str__(self):
        return "<{}>".format(self.__class__.__name__)
