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
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import stats
from scipy import interpolate


class DamgDepFragModelsNoC():

    # default im arrays for plots and outputs
    x_ims_g = np.logspace(np.log10(1e-3), np.log10(10), 100)
    x_ims_ms2 = x_ims_g*9.81
    
    def __init__(self, psdm, dsc, sigmab2b=0., imt="IM"):
        self.imt = imt
        self.psdm = psdm
        self.dsc = dsc
        # get rid of cases where there is collapse, and edp==0 (with tol)
        # filt = np.logical_and(psdm.nocoll_g2, psdm.ds_g2 != 0, edp > 1e-5)
        self.filt_g1 = np.logical_and(psdm.nocoll, psdm.nonz_en_g1)
        self.filt_g12 = np.logical_and(psdm.nocoll, psdm.hysts_g12 > psdm.ZERO_TOL)
        # edp, maxds_g1 and ims_g2 for fragility
        self._edp = psdm.hysts_g12[self.filt_g12]
        self._maxds_g1 = psdm.maxds_g1[self.filt_g12]
        self._ims_g2 = psdm.ims_g2[self.filt_g12]
        self.__edp = psdm.hysts_g1[self.filt_g1] # psdm.hysts_g1[filt_g1]
        self.__maxds_g1 = np.zeros_like(psdm.maxds_g1[self.filt_g1]) # psdm.maxds_g1[filt_g1]
        self.__ims_g2 = psdm.ims_g1[self.filt_g1] # np.zeros_like(psdm.ims_g1[filt_g1])
        self.edp = np.concatenate([self._edp, self.__edp])
        self.maxds_g1 = np.concatenate([self._maxds_g1, self.__maxds_g1])
        self.ims_g2 = np.concatenate([self._ims_g2, self.__ims_g2])
        # evaluate psdm
        psd = psdm(self.maxds_g1, self.ims_g2)

        # sigma, residuals and beta
        self.sigma_psdm = np.sqrt(np.sum((np.log(self.edp) - np.log(psd))**2) / (len(self.edp)-2))
        self.residuals = psd - self.edp
        self.residuals_log = np.log(self.edp) - np.log(psd)
        # compute beta accounting for building-to-building variability 
        # see Martins and Silva (2020)
        self.beta = np.sqrt( self.sigma_psdm**2 + sigmab2b**2 ) / psdm.d
        # hysteretic energy given damage state thresholds
        self.E_ds = dict()
        for ds in range(0,dsc.get_num_ds()+1):
            self.E_ds[ds] = psdm.a * dsc.get_ds_thresh(ds) ** psdm.b
        # median values of fragility curves for DS-g2|DS-g1 (DS-g1==0, mainshock fragility)
        self.mus = dict()
        for ds1 in range(0,dsc.get_num_ds()+1):
            for ds2 in range(0,dsc.get_num_ds()+1):
                if ds2 > ds1:
                    self.mus[(ds1,ds2)] = ((self.E_ds[ds2] - self.E_ds[ds1]) / \
                           (psdm.c0*(1 - psdm.m * dsc.get_ds_thresh(ds1)))) ** (1/psdm.d)

    
    def get_frag_params(self, ds2, ds1=0, unit="m/s2"):
        '''
        ds1 = 0 returns the mainshock fragility
        ds1 != 0 returns the aftershock fragility for ds2
        this gets overridden in DamgDepFragModels
        '''
        if ds2 == 0:
            raise Exception("check ds2, it cannot be zero")
        if unit == "g":
            return self.mus[(ds1, ds2)]/9.81, self.beta
        return self.mus[(ds1, ds2)], self.beta
    
    
    def get_frags_params_df(self, unit="m/s2"):
        data = dict()
        for (ds1, ds2) in self.mus.keys():
            mu, beta = self.get_frag_params(ds2, ds1, unit)
            if ds1 == 0:
                data["DS{}".format(ds2)] = dict(mu=mu, beta=beta)
            else:
                data["DS{}|DS{}".format(ds2, ds1)] = dict(mu=mu, beta=beta)
        return pd.DataFrame(data).T
    
    
    def get_ims(self, ims=None, unit="m/s2"):
        if ims is None:
            if unit == "g":
                ims = self.x_ims_g
            else:
                ims = self.x_ims_ms2
        return ims

    
    def get_fragility(self, ims, ds2, ds1=0, unit="m/s2"):
        '''
        this gets overridden in DamgDepFragModels
        '''
        mu, beta = self.get_frag_params(ds2, ds1, unit)
        return norm.cdf((np.log(ims) - np.log(mu))/beta)


    def get_fragilities(self, ims=None, unit="m/s2"):
        ims = self.get_ims(ims, unit)
        P_ds = dict()
        for (ds1, ds2) in self.mus.keys():
            P_ds[(ds1, ds2)] = self.get_fragility(ims, ds2, ds1, unit)
        return P_ds
    
    
    def get_fragilities_df(self, ims=None, unit="g"):
        ims = self.get_ims(ims, unit)
        P_ds = self.get_fragilities(ims, unit)
        data = {"{} ({})".format(self.imt, unit): ims}
        for (ds1, ds2) in P_ds.keys():
            if ds1 == 0:
                label = "DS"+str(ds2)
            else:
                label = "DS"+str(ds2)+"|DS"+str(ds1)
            data[label] = P_ds[(ds1, ds2)]
        return pd.DataFrame(data)
        

    def plot_frag_all(self, unit="g", imt=None, x_ims=None, max_img=None):
        x_ims = self.get_ims(x_ims, unit)
        P_ds = self.get_fragilities(x_ims, unit)
        
        fig, ax = plt.subplots(figsize=(6,6))
        for (ds1, ds2) in P_ds.keys():
            if ds1 == 0:
                label = "DS"+str(ds2)
                ls = '-'
            else:
                label = "DS"+str(ds2)+"|DS"+str(ds1)
                ls = ["--", "-.", ":"][ds1-1]
            color = ["g", "y", [1.,0.6,0.], "r"][ds2-1]
            ax.plot(x_ims, P_ds[(ds1, ds2)], lw=1, ls=ls, color=color, label=label)
        if imt is None:
            imt = self.imt
        ax.set_xlabel('{} ({})'.format(imt, unit))
        ax.set_ylabel('P(DS-G2 >= ds | IM, DS-G1)')
        ax.legend(framealpha=0.5)
        ax.set_ylim(0.,1.)
        ax.set_xlim(xmin=0.)
        if max_img is not None:
            ax.set_xlim(xmax=max_img)
        return fig, ax


    def plot_frag_ds(self, ds1_plot=0, unit="g", imt=None, max_img=None):
        '''
        ds1==0, i.e., mainshock fragility
        '''
        x_ims = self.get_ims(None, unit)
        P_ds = self.get_fragilities(x_ims, unit)
        
        fig, ax = plt.subplots(figsize=(6,6))
        for (ds1, ds2) in P_ds.keys():
            if ds1 == ds1_plot:
                if ds1 == 0:
                    label = "DS"+str(ds2)
                    ls = '-'
                else:
                    label = "DS"+str(ds2)+"|DS"+str(ds1)
                    ls = ["--", "-.", ":"][ds1-1]
                color = ["b", "m", "g", "r"][ds2-1]
                ax.plot(x_ims, P_ds[(ds1, ds2)], lw=1, ls=ls, color=color, label=label)
        if imt is None:
            imt = self.imt
        ax.set_xlabel('{} ({})'.format(imt, unit))
        ax.set_ylabel('P(DS-G2 >= ds | IM, DS-G1)')
        ax.legend(framealpha=0.5)
        ax.set_ylim(0.,1.)
        ax.set_xlim(xmin=0.)
        if max_img is not None:
            ax.set_xlim(xmax=max_img)
        return ax

    
    def check_plots(self, unit="m/s2", imt=None, save=False, path=None):
        fig, ax = self.plot_frag_all(unit, imt)
        if save:
            fig.savefig(os.path.join(path, "frag_01.png"),
                        bbox_inches='tight', dpi=600, format="png")

        plt.figure()
        plt.scatter(self.psdm.ims_g1, self.psdm.maxds_g1)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('IM G1')
        plt.ylabel('Max drift G1')
        if save:
            plt.savefig(os.path.join(path, "frag_02.png"),
                        bbox_inches='tight', dpi=600, format="png")
        
        # histogram residuals [:self._maxds_g1.shape[0]]
        mu, std = stats.norm.fit(self.residuals_log) # Fit a normal distribution to the data
        _, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        h, vals = np.histogram(self.residuals_log, bins=100, density=True)
        f = interpolate.interp1d(np.cumsum(h)*np.diff(vals), vals[:-1]+np.diff(vals)/2)
        ax1.bar(vals[:-1]+np.diff(vals)/2, h, width=0.9*min(np.diff(vals)))
        xmin, xmax = ax1.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        ax1.plot(x, stats.norm.pdf(x, mu, std), 'k', linewidth=2)
        ax1.set_ylabel("PDF")
        ax2.plot(x, stats.norm.cdf(x, mu, std), 'k', linewidth=2)
        ax2.plot(vals[:-1]+np.diff(vals)/2, np.cumsum(h)*np.diff(vals))
        ax2.set_xlabel("log-space residuals")
        ax2.set_ylabel("CDF")
        if save:
            plt.savefig(os.path.join(path, "frag_03.png"),
                        bbox_inches='tight', dpi=600, format="png")

        
        # histogram residuals [:self._maxds_g1.shape[0]]
        mu, std = stats.norm.fit(self.residuals) # Fit a normal distribution to the data
        _, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        h, vals = np.histogram(self.residuals, bins=100, density=True)
        ax1.bar(vals[:-1]+np.diff(vals)/2, h, width=0.9*min(np.diff(vals)))
        xmin, xmax = ax1.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        ax1.plot(x, stats.norm.pdf(x, mu, std), 'k', linewidth=2)
        ax1.set_ylabel("PDF")
        ax2.plot(x, stats.norm.cdf(x, mu, std), 'k', linewidth=2)
        ax2.plot(vals[:-1]+np.diff(vals)/2, np.cumsum(h)*np.diff(vals))
        ax2.set_xlabel("lin-space residuals")
        ax2.set_ylabel("CDF")
        if save:
            plt.savefig(os.path.join(path, "frag_04.png"),
                        bbox_inches='tight', dpi=600, format="png")


        # cloud of points 3d hysteretic energy GM1+GM2 vs max drift GM1 vs IMs GM2
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        X = np.linspace(0, np.max(self._ims_g2), 20)
        Y = np.linspace(0, np.max(self._maxds_g1), 20)
        X, Y = np.meshgrid(X, Y)
        R = self.psdm(Y.flatten(), X.flatten())
        Z = R.reshape(X.shape)
        ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0,
                        antialiased=False, alpha=0.5)
        ax.scatter(self._ims_g2, self._maxds_g1, self._edp, s=2, color="m")
        ax.scatter(self.__ims_g2, self.__maxds_g1, self.__edp, s=2, color="b")
        xs = np.linspace(0., np.max(self.maxds_g1), 100)
        ax.plot([0.]*len(xs), xs, self.psdm.power_law(xs, self.psdm.a, self.psdm.b),
                color="k", lw=2, zorder=200)
        xs = np.linspace(0., np.max(self._ims_g2), 100)
        ax.plot(xs, [0.]*len(xs), self.psdm.power_law(xs, self.psdm.c0, self.psdm.d),
                color="k", lw=2, zorder=200)
        ds_states = [0,1,2,3,4]
        m = cm.ScalarMappable(cmap=cm.jet)
        m.set_array(ds_states)
        ds_colors = m.to_rgba(ds_states) #["c", "g", "m", "b", "r"]
        for ds, col in zip(ds_states[1:], ds_colors[1:]):
            ys = self.dsc.get_ds_thresh(ds)*np.ones_like(xs)            
            ax.plot(xs, ys, self.psdm(ys, xs),
                    color=col, label="DS"+str(ds))
        ax.set_xlabel('IM G2')
        ax.set_ylabel('Max drift G1')
        ax.set_zlabel('Total Hysteretic Energy')
        ax.view_init(elev=25, azim=-140)
        if save:
            fig.savefig(os.path.join(path, "frag_05.png"),
                        bbox_inches='tight', dpi=600, format="png")
        else:
            plt.show()


    def __repr__(self):
        return self.__str__()
        
    def __str__(self):
        return "<{}>".format(self.__class__.__name__)


