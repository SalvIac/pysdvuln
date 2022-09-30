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
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import cm


class PSDM_gg21():

    def __init__(self, ims_g1, ims_g2, hysts_g1, hysts_g2, maxds_g1, 
                 dam_state_cl, maxds=None, collapses=None, zero_tol=1e-3,
                 unit="m/s2", mode="modified", **kwargs):
        # this is to make sure acceleration is in m/s2
        if unit == "g":
            scale = 1/9.81
        elif unit == "m/s2":
            scale = 1.
        else:
            raise Exception("unit can only be 'g' or 'm/s2'")
        self.__dict__.update(kwargs)
        self.ims_g1 = np.abs(ims_g1)/scale
        self.ims_g2 = np.abs(ims_g2)/scale
        self.hysts_g1 = np.abs(hysts_g1)
        self.hysts_g2 = np.abs(hysts_g2)
        self.hysts_g12 = self.hysts_g1 + self.hysts_g2
        self.maxds_g1 = np.abs(maxds_g1)
        self.dam_state_cl = dam_state_cl
        if maxds is None and collapses is None:
            raise Exception("specify maxds or collapses")
        if collapses is not None:
            self.collapses = collapses
        if maxds is not None:
            self.maxds = np.abs(maxds)
        self.ZERO_TOL = zero_tol # tolerance for zero hysteretic energy
        self.COLLAPSE_FACTOR = dam_state_cl.COLLAPSE_FACTOR
        self.mode = mode
        np.random.seed(42) # for reproducibility
        self.calculate()
            
        
    def calculate(self):
        # classify GM1
        self.ds_g1 = self.dam_state_cl.classify(self.maxds_g1)
        # self.ds_g2 = np.clip(dam_state_cl.classify(self.maxds_g2), self.ds_g1, None)
        # filter for "collapse" cases for mainshock
        if "collapses" in self.__dict__.keys():
            self.nocoll = ~self.collapses
        else:
            self.nocoll = ~self.check_collapse(self.maxds)
        # 1. Using the MS data only (𝜃𝑀𝑆, 𝐸𝐻,𝑀𝑆), the relationship 𝑎𝜃𝑏
        # 𝑀𝑆 is fitted. The parameters 𝑎 and 𝑏 are estimated via the
        # linear least squares method in the log-log space.
        # filter zero hysteretic energy
        self.nonz_en_g1 = self.hysts_g1 > self.ZERO_TOL
        filt_g1 = np.logical_and(self.nocoll, self.nonz_en_g1)
        # (a, b), pcov = curve_fit(self.power_law,
        #                          self.maxds_g1[filt_g1],
        #                          self.hysts_g1[filt_g1])
        # (a, b), pcov = curve_fit(self.linear_model,
        #                           np.log(self.maxds_g1[filt_g1]),
        #                           np.log(self.hysts_g1[filt_g1]))
        # a = np.exp(a)
        reg = LinearRegression().fit(np.log(self.maxds_g1[filt_g1]).reshape(-1, 1),
                                     np.log(self.hysts_g1[filt_g1]))
        a = np.exp(reg.intercept_)
        b = reg.coef_[0]
        # e MS data only (𝐼𝑀𝑀𝑆, 𝐸𝐻,𝑀𝑆), the relationship 𝑐0𝐼𝑀𝑑
        # 𝐴𝑆 is fitted. The parameters 𝑐0 and 𝑑 are estimated as per point 1.
        # (c0, d), pcov = curve_fit(self.power_law, 
        #                           self.ims_g1[filt_g1],
        #                           self.hysts_g1[filt_g1])
        # (c0, d), pcov = curve_fit(self.linear_model,
        #                           np.log(self.ims_g1[filt_g1]),
        #                           np.log(self.hysts_g1[filt_g1]))
        # c0 = np.exp(c0)
        reg = LinearRegression().fit(np.log(self.ims_g1[filt_g1]).reshape(-1, 1),
                                     np.log(self.hysts_g1[filt_g1]))
        c0 = np.exp(reg.intercept_)
        d = reg.coef_[0]
        # 3. Using the AS data (𝜃𝑀𝑆, 𝐼 𝑀𝐴𝑆, 𝐸𝐻,𝐴𝑆), the parameter 𝑚 is
        # estimated via the nonlinear least squares to the function
        # 𝐸𝐻,𝐴𝑆 = (1−𝑚𝜃𝑀𝑆) 𝑐0𝐼𝑀𝑑𝐴𝑆.
        # filter zero hysteretic energy
        self.nonz_en_g2 = self.hysts_g2 > self.ZERO_TOL
        filt_g2 = np.logical_and(self.nocoll, self.nonz_en_g2)
        if self.mode == "original":
            m, pcov = curve_fit(lambda x, m: self.power_law_gm2_2d(x, c0, d, m),
                                np.vstack([self.maxds_g1[filt_g2], 
                                            self.ims_g2[filt_g2]]).T,
                                self.hysts_g2[filt_g2])
        elif self.mode == "modified":
            m, pcov = curve_fit(lambda x, m: self.log_power_law_gm2_2d(x, c0, d, m),
                                np.vstack([self.maxds_g1[filt_g2], 
                                            self.ims_g2[filt_g2]]).T,
                                np.log(self.hysts_g2[filt_g2]), p0=0.,
                                bounds=(-np.inf, np.max(1/self.maxds_g1[filt_g2])))
        else:
            raise Exception("check mode")
        self.a = a
        self.b = b
        self.c0 = c0
        self.d = d
        self.m = m[0]
    
    
    def check_collapse(self, drift):
        return self.dam_state_cl.check_collapse(drift)
    
    
    @classmethod
    def from_analyses(cls, analyses, dam_state_cl, zero_tol=1e-3):
        ims_g1 = list()
        ims_g2 = list()
        hysts_g1 = list()
        hysts_g2 = list()
        maxds_g1 = list()
        maxds = list()
        for an in analyses:
            ims_g1.append( an["im_g1"] )
            ims_g2.append( an["im_g2"] )
            rt = an["rp"].end_rest_time
            hysts_g1.append( an["opr"].calculate_hyst_energy( end=rt ) )
            hysts_g2.append( an["opr"].calculate_hyst_energy( start=rt ) )
            # check hysteretic energy g1+g2 ~= g12
            if np.abs(an["opr"].calculate_hyst_energy() - (hysts_g1[-1]+hysts_g2[-1])) > 1e-5:
                print(hysts_g1[-1], hysts_g2[-1], hysts_g1[-1]+hysts_g2[-1],
                      an["opr"].calculate_hyst_energy(),
                      an["opr"].calculate_hyst_energy() - (hysts_g1[-1]+hysts_g2[-1]))
                raise Exception("something wrong with hysts_g1 and hysts_g2")
            maxds_g1.append( an["opr"].get_max_disp( end=rt ) )
            maxds.append( an["opr"].get_max_disp() )
        return cls(ims_g1, ims_g2, hysts_g1, hysts_g2, maxds_g1, dam_state_cl, 
                   maxds=maxds, zero_tol=zero_tol)


    @staticmethod
    def extract_lists(rp, opr, im_g1, im_g2):
        rt = rp.end_rest_time
        hyst_g1 = opr.calculate_hyst_energy( end=rt )
        hyst_g2 = opr.calculate_hyst_energy( start=rt )
        # check hysteretic energy g1+g2 ~= g12
        if np.abs(opr.calculate_hyst_energy() - (hyst_g1+hyst_g2)) > 1e-5:
            print(hyst_g1, hyst_g2, hyst_g1+hyst_g2,
                  opr.calculate_hyst_energy(),
                  opr.calculate_hyst_energy() - (hyst_g1+hyst_g2))
            raise Exception("something wrong with hysts_g1 and hysts_g2")
        else:
            hyst_g12 = hyst_g1+hyst_g2
        maxd_g1 = opr.get_max_disp( end=rt )
        maxd = opr.get_max_disp()
        return dict(im_g1=im_g1, im_g2=im_g2, 
                    hyst_g1=hyst_g1, hyst_g2=hyst_g2, hyst_g12=hyst_g12,
                    maxd_g1=maxd_g1, maxd=maxd)

    
    @classmethod
    def power_law(cls, x, a, b):
        # simple power law model
        return a*np.power(x, b)

    @classmethod
    def linear_model(cls, x, a, b):
        return a + b*x

    @classmethod
    def power_law_2d(cls, data, a, b, c0, d, m):
        '''
        proposed by Gentile and Galasso (2021)
        data: 2d numpy array with col1 gm1_max_drift and col2 gm2_ims
        a, b, c0, d, m are 5 parameters
        '''
        return cls.power_law_gm1(data[:,0], a, b) + \
               cls.power_law_gm2_2d(data, c0, d, m)
    
    @classmethod
    def power_law_gm1(cls, gm1_max_drift, a, b):
        # to model hysteretic energy for gm1 (e.g., a mainshock)
        return a * np.power(gm1_max_drift, b)
        
    @classmethod
    def power_law_gm2_2d(cls, data, c0, d, m):
        # to model hysteretic energy for gm2 (e.g., an aftershock)
        # data: 2d numpy array with col1 gm1_max_drift and col2 gm2_ims
        gm1_max_drift = data[:,0]
        gm2_ims = data[:,1]
        return (1 - m * gm1_max_drift) * c0 * np.power(gm2_ims, d)


    @classmethod
    def log_power_law_gm2_2d(cls, data, c0, d, m):
        # log of power_law_gm2_2d
        gm1_max_drift = data[:,0]
        gm2_ims = data[:,1]
        return np.log(1 - m * gm1_max_drift) + np.log(c0) + d*np.log(gm2_ims)


    def get_params(self, form="dict"):
        if form == "dict":
            return {"a": self.a,
                    "b": self.b,
                    "c0": self.c0,
                    "d": self.d,
                    "m": self.m}
        else:
            return self.a, self.b, self.c0, self.d, self.m
    

    def get_params_df(self):
        return pd.DataFrame(self.get_params("dict"), index=[0]).T
    

    def get_ds_colors(self):
        return self.dam_state_cl.get_ds_colors()


    def check_plots(self, unit="g", imt="IM", save=False, path=""):
        '''
        these plots are similar to the ones in Gentile and Galasso (2021)
        '''
        if unit == "g":
            scale = 1/9.81
        else:
            scale = 1.
        filt_g1 = np.logical_and(self.nocoll, self.nonz_en_g1)
        filt_g2 = np.logical_and(self.nocoll, self.nonz_en_g2)
        ds_states, ds_colors = self.get_ds_colors()
        
        # plot IM-gm1 and IM-gm2 scatter 2d, colors depends on status after GM1
        fig, ax = plt.subplots(figsize=(6,6))
        ax.scatter(self.ims_g1[~self.nocoll]*scale,
                   self.ims_g2[~self.nocoll]*scale,
                   s=20, lw=0.5, marker="^",
                   color="k", label="Collapsed")
        ax.scatter(self.ims_g1[~self.nonz_en_g1]*scale,
                   self.ims_g2[~self.nonz_en_g1]*scale,
                   s=20, lw=0.5, marker="*",
                   color="k", label="G1 Hyst=0")
        for ds, col in zip(ds_states, ds_colors):
            ax.scatter(self.ims_g1[filt_g1][self.ds_g1[filt_g1]==ds]*scale, 
                       self.ims_g2[filt_g1][self.ds_g1[filt_g1]==ds]*scale,
                       s=20, lw=0.5, marker="o",
                       color=col, label="G1 DS"+str(ds))
        ax.legend(framealpha=0.5)
        ax.set_xlabel(r'{} G1 ({})'.format(imt, unit))
        ax.set_ylabel(r'{} G2 ({})'.format(imt, unit))
        if save:
            fig.savefig(os.path.join(path, "psdm_01.png"),
                        bbox_inches='tight', dpi=600, format="png")
        

        # # plot IM-gm1, IM-gm2 and hyst12 scatter 3d
        # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        # ax.scatter(self.ims_g1[filt_g1], self.ims_g2[filt_g1], 
        #            self.hysts_g12[filt_g1],
        #            s=10, lw=0.5, marker="o",
        #            color="k", label="Collapsed or hyst=0")
        # for ds, col in zip(ds_states, ds_colors):
        #     ax.scatter(self.ims_g1[self.ds_g1==ds], 
        #                self.ims_g2[self.ds_g1==ds],
        #                self.hysts_g12[self.ds_g1==ds],
        #                s=5, lw=0.5, marker="o",
        #                color=col, label="DS"+str(ds))
        # ax.legend(framealpha=0.5)
        # ax.set_xlabel(r'{} G1 ({})'.format(imt, unit))
        # ax.set_ylabel(r'{} G2 ({})'.format(imt, unit))
        # ax.set_zlabel('Total Hysteretic Energy')


        # 3d collapse / no collapse vs max drift GM1 vs IMs GM2
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.scatter(self.ims_g2[~self.nocoll]*scale,
                   self.maxds_g1[~self.nocoll], 1,
                   s=20, lw=0.5, color="r", label="Collapsed")
        ax.scatter(self.ims_g2[self.nocoll]*scale,
                   self.maxds_g1[self.nocoll], 0.,
                   s=20, lw=0.5, color="c", label="Not collapsed")
        ax.legend(framealpha=0.5)
        ax.set_xlabel('{} G2 ({})'.format(imt, unit))
        ax.set_ylabel('Max drift G1')
        ax.set_zlabel('Collapse')
        ax.view_init(elev=25, azim=-140)
        if save:
            fig.savefig(os.path.join(path, "psdm_02.png"),
                        bbox_inches='tight', dpi=600, format="png")
        

        # 2d hysteretic energy vs max drift ground motion 1 (mainshock)
        fig, ax = plt.subplots()
        for ds, col in zip(ds_states, ds_colors):
            ax.scatter(self.maxds_g1[filt_g1][self.ds_g1[filt_g1]==ds],
                       self.hysts_g1[filt_g1][self.ds_g1[filt_g1]==ds],
                       s=2, color=col, label="DS"+str(ds))
        xs = np.linspace(0., np.max(self.maxds_g1[filt_g1]), 100)
        ax.plot(xs, self.power_law(xs, self.a, self.b), color="k")
        ax.set_xlabel('Max drift G1')
        ax.set_ylabel('Hysteretic Energy G1')
        ax.legend(framealpha=0.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        if save:
            fig.savefig(os.path.join(path, "psdm_03.png"),
                        bbox_inches='tight', dpi=600, format="png")


        # 3d hysteretic energy vs max drift ground motion 1 (mainshock)
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        for ds, col in zip(ds_states, ds_colors):
            ax.scatter([0.]*len(self.ims_g1[filt_g1][self.ds_g1[filt_g1]==ds]),
                       self.maxds_g1[filt_g1][self.ds_g1[filt_g1]==ds],
                       self.hysts_g1[filt_g1][self.ds_g1[filt_g1]==ds],
                       s=2, color=col, label="DS"+str(ds))
        xs = np.linspace(0., np.max(self.maxds_g1[filt_g1]), 100)
        ax.plot([0.]*len(xs), xs, self.power_law(xs, self.a, self.b),
                color="k")
        ax.set_xlabel('{} G2 ({})'.format(imt, unit)) # (m/s2)
        ax.set_ylabel('Max drift G1') # (m)
        ax.set_zlabel('Hysteretic Energy G1') # (Nm)
        ax.set_xlim([0, max(self.ims_g2)*scale])
        ax.set_ylim([0, np.max(self.maxds_g1[filt_g1])])
        ax.legend(framealpha=0.5)
        ax.view_init(elev=25, azim=-140)
        if save:
            fig.savefig(os.path.join(path, "psdm_04.png"),
                        bbox_inches='tight', dpi=600, format="png")


        # 2d hysteretic energy vs max drift ground motion 2 (aftershock)
        fig, ax = plt.subplots()
        for ds, col in zip([0,1,2,3,4], ds_colors):
            ax.scatter(self.ims_g1[filt_g1][self.ds_g1[filt_g1]==ds]*scale,
                       self.hysts_g1[filt_g1][self.ds_g1[filt_g1]==ds],
                       s=2, color=col, label="DS"+str(ds))
        xs = np.linspace(0., np.max(self.ims_g1[filt_g1]), 100)
        ax.plot(xs*scale, self.power_law(xs, self.c0, self.d), color="k")
        ax.set_xlabel('{} G1 ({})'.format(imt, unit)) # (m/s2)
        ax.set_ylabel('Hysteretic Energy G1') # (Nm)
        ax.legend(framealpha=0.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        if save:
            fig.savefig(os.path.join(path, "psdm_05.png"),
                        bbox_inches='tight', dpi=600, format="png")


        # 3d hysteretic energy GM2 vs max drift GM1 vs IMs GM2 (aftershock)
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        for ds, col in zip(ds_states, ds_colors):
            ax.scatter(self.ims_g1[filt_g1][self.ds_g1[filt_g1]==ds]*scale,
                       np.zeros_like(self.maxds_g1[filt_g1][self.ds_g1[filt_g1]==ds]),
                       self.hysts_g1[filt_g1][self.ds_g1[filt_g1]==ds],
                       s=2, color=col, label="DS"+str(ds))
        ax.scatter(self.ims_g2[filt_g2]*scale,
                   self.maxds_g1[filt_g2],
                   self.hysts_g2[filt_g2],
                   s=0.1, color="m")
        xs = np.linspace(0., np.max(self.ims_g1[filt_g1]), 100)
        ax.plot(xs*scale, [0.]*len(xs), self.power_law(xs, self.c0, self.d),
                color="k")
        ax.set_xlabel('{} G2 ({})'.format(imt, unit)) # (m/s2)
        ax.set_ylabel('Max drift G1') # (m)
        ax.set_zlabel('Hysteretic Energy G2') # (Nm)
        ax.set_xlim([0, max(self.ims_g2)*scale])
        ax.set_ylim([0, np.max(self.maxds_g1[filt_g1])])
        ax.view_init(elev=25, azim=-140)
        if save:
            fig.savefig(os.path.join(path, "psdm_06.png"),
                        bbox_inches='tight', dpi=600, format="png")

        self.plot(unit, imt, save, path)
        
  
    
    def plot(self, unit="g", imt="IM", save=False, path=""):
        if unit == "g":
            scale = 1/9.81
        else:
            scale = 1.
        filt_g1 = np.logical_and(self.nocoll, self.nonz_en_g1)
        filt_g2 = np.logical_and(self.nocoll, self.nonz_en_g2)
        ds_states, ds_colors = self.get_ds_colors()

        # 3d hysteretic energy GM1+GM2 vs max drift GM1 vs IMs GM2
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        X = np.linspace(0, np.max(self.ims_g2[filt_g2]), 20)
        Y = np.linspace(0, np.max(self.maxds_g1[filt_g2]), 20)
        X, Y = np.meshgrid(X, Y)
        R = self(Y.flatten(), X.flatten())
        Z = R.reshape(X.shape)
        ax.scatter(self.ims_g2[self.nocoll]*scale, self.maxds_g1[self.nocoll],
                   self.hysts_g12[self.nocoll], s=2, color="m")
        # ax.scatter(self.ims_g2, self.maxds_g1, self.hysts_g12, s=2, color="g")
        ax.plot_surface(X*scale, Y, Z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False, alpha=0.5)
        xs = np.linspace(0., np.max(self.maxds_g1[filt_g2]), 100)
        ax.plot([0.]*len(xs), xs, self.power_law(xs, self.a, self.b),
                color="k", lw=2, zorder=200)
        xs = np.linspace(0., np.max(self.ims_g2[filt_g2]), 100)
        ax.plot(xs*scale, [0.]*len(xs), self.power_law(xs, self.c0, self.d),
                color="k", lw=2, zorder=200)
        ax.set_xlabel('{} G2 ({})'.format(imt, unit))
        ax.set_ylabel('Max drift G1')
        ax.set_zlabel('Total Hysteretic Energy')
        ax.view_init(elev=25, azim=-140)
        if save:
            fig.savefig(os.path.join(path, "psdm.png"),
                        bbox_inches='tight', dpi=600, format="png")
        else:
            plt.show()
    

    def __call__(self, gm1_max_drift, gm2_ims):
        return self.power_law_2d(np.vstack([gm1_max_drift, gm2_ims]).T,
                                 self.a, self.b, self.c0, self.d, self.m)
    
    
    def __repr__(self):
        return self.__str__()
        
    def __str__(self):
        return "<{}>".format(self.__class__.__name__)
    
    
    #%%
    
    # param, param_cov = curve_fit(power_law, maxds, hysts)
    # x = np.linspace(0., max(maxds))
    # y = power_law(x, param[0], param[1])
    
    # fig = plt.figure(figsize=(6,6))
    # plt.scatter(maxds, hysts, s=5, edgecolor='k', linewidth=0.1)
    # plt.plot(x, y, color='k', linewidth=1)
    # plt.xlabel("max displacement")
    # plt.ylabel("hysteretic energy")
    # plt.show()


    # filt_g1 = np.logical_and(psdm.nocoll_g1, psdm.nonz_en_g1)
    
    # # benchmark linear fit in a log-log space
    # (a, b), pcov = curve_fit(psdm.linear_model,
    #                          np.log(psdm.maxds_g1[filt_g1]),
    #                          np.log(psdm.hysts_g1[filt_g1]))
    # a = np.exp(a)
    # print(a,b)

    # # this fits a log power law where x is given linearly and y is log
    # # this gives the same result as the benchmark because the residuals are 
    # # assumed gaussian in a log space
    # def log_power_law(x, a, b):
    #     return np.log(a*np.power(x, b))
    # (a, b), pcov = curve_fit(log_power_law,
    #                          psdm.maxds_g1[filt_g1],
    #                          np.log(psdm.hysts_g1[filt_g1]))
    # print(a,b)
    
    # # this gives a totally different result because the residuals are 
    # # assumed gaussian in a lin space
    # (a, b), pcov = curve_fit(psdm.power_law,
    #                          psdm.maxds_g1[filt_g1],
    #                          psdm.hysts_g1[filt_g1])
    # print(a,b)

