# -*- coding: utf-8 -*-
"""
@author: Salvatore
"""

import os
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator
import matplotlib.pyplot as plt
from matplotlib import cm


class Vuln3d():
    
    def __init__(self, ddvc):
        self.ddvc = ddvc
        points = list()
        z = list()
        # mainshock-only
        ims = ddvc.get_ims(unit="g")
        vuln = ddvc.get_vuln_curve_ds1(ds1=0)
        for i1, i2, i3 in zip(ims, [0.]*len(ims), vuln):
            points.append( [i1,i2] )
            z.append(i3)
        for i1, i2, i3 in zip([0.]*len(ims), ims, vuln):
            points.append( [i1,i2] )
            z.append(i3)
        # state-dependent
        for ds1 in range(1, len(ddvc.vulns.keys())):
            vuln_ds = ddvc.get_vuln_curve_ds1(ds1=ds1)
            # index where first point vuln_ds meets vuln mainshock
            ind = np.searchsorted(vuln, vuln_ds[0])
            # linear interpolation
            _ims = ims[ind-1:ind+1]
            _vuln = vuln[ind-1:ind+1]
            _ims_inter = _ims[0] + np.diff(_ims)/np.diff(_vuln) * (vuln_ds[0]-_vuln[0])
            for i1, i2, i3 in zip(_ims_inter*np.ones(len(ims)), ims, vuln_ds):
                points.append( [i1,i2] )
                z.append(i3)
        # # avoid weird interpolation after vuln curve given ds3
        # # this scales up the minimum of ds3 curve according to mainshock vuln
        # ind1 = np.searchsorted(vuln, 0.99)
        # _ind1 = np.linspace(ind, ind1, 10, dtype=int)
        # for i in _ind1[1:]:
        #     im = ims[i]
        #     # min max scaler
        #     X_std = (vuln_ds - vuln_ds.min()) / (vuln_ds.max() - vuln_ds.min())
        #     _min = vuln[i]
        #     vuln_scaled = X_std * (1. - _min) + _min
        #     for i1, i2, i3 in zip(im*np.ones(len(ims)), ims, vuln_scaled):
        #         points.append( [i1,i2] )
        #         z.append(i3)
        # store values and interpolate
        self.X = ims #np.linspace(0.05, 3.1, 100)
        self.Y = ims #np.linspace(0.05, 3.1, 100)
        X, Y = np.meshgrid(self.X, self.Y)
        interp = LinearNDInterpolator(np.array(points, dtype=float), z)
        self.Z = interp(X, Y)
        self.Z[np.isnan(self.Z)] = 1.
        self.Z[self.Z>1] = 1.
            
    
    def get_vuln_3d_df(self):
        return pd.DataFrame(data=self.Z, index=self.X, columns=self.Y)
    

    def plot_vuln_surf(self, unit="g", imt="IM", save=False, path="",
                       max_img=None):
        X, Y = np.meshgrid(self.X, self.Y)
        if max_img is None:
            max_img = np.max(X)+1e-2
        cond = np.logical_and(X <= max_img, Y <= max_img)
        inds = np.where(cond)
        n = int(np.sqrt(np.sum(cond)))

        # 3d vulnerability surface vs GM1 and GM2
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ims = self.X
        vuln = self.ddvc.get_vuln_curve_ds1(ds1=0)
        ax.plot(ims[ims<=max_img],
                [0.]*len(ims[ims<=max_img]),
                vuln[ims<=max_img], color="k", lw=2, zorder=200)
        ax.plot([0.]*len(ims[ims<=max_img]),
                ims[ims<=max_img],
                vuln[ims<=max_img], color="k", lw=2, zorder=200)
        for ds1 in range(1,len(self.ddvc.vulns.keys())):
            vuln_ds = self.ddvc.get_vuln_curve_ds1(ds1=ds1)
            ind = np.searchsorted(vuln, vuln_ds[0])
            _ims = ims[ind-1:ind+1]
            _vuln = vuln[ind-1:ind+1]
            _ims_inter = _ims[0] + np.diff(_ims)/np.diff(_vuln) * (vuln_ds[0]-_vuln[0])
            ax.plot(_ims_inter*np.ones(len(ims[ims<=max_img])),
                    ims[ims<=max_img], vuln_ds[ims<=max_img],
                    color="k", lw=2, zorder=200)
        # # avoid weird interpolation after vuln curve given ds3
        # ind1 = np.searchsorted(vuln, 0.99)
        # _ind1 = np.linspace(ind, ind1, 10, dtype=int)
        # for i in _ind1[1:]:
        #     im = ims[i]
        #     # min max scaler
        #     X_std = (vuln_ds - vuln_ds.min()) / (vuln_ds.max() - vuln_ds.min())
        #     _min = vuln[i]
        #     vuln_scaled = X_std * (1. - _min) + _min
        #     ax.plot([im]*len(ims[ims<=max_img]), ims[ims<=max_img],
        #             vuln_scaled[ims<=max_img], color="k", lw=2, zorder=200)
        surf = ax.plot_surface(X[inds[0], inds[1]].reshape(n,n),
                               Y[inds[0], inds[1]].reshape(n,n),
                               self.Z[inds[0], inds[1]].reshape(n,n),
                               cmap=cm.coolwarm, linewidth=0,
                               antialiased=False, alpha=0.5)
        ax.contour(X[inds[0], inds[1]].reshape(n,n),
                   Y[inds[0], inds[1]].reshape(n,n),
                   self.Z[inds[0], inds[1]].reshape(n,n),
                   zdir='z', offset=0.01,
                   cmap=cm.coolwarm)
        ax.set_xlabel('{} G1 ({})'.format(imt, unit))
        ax.set_ylabel('{} G2 ({})'.format(imt, unit))
        ax.set_zlabel("$E(LR | IM_{G1}, IM_{G2})$") #'Loss Ratio')
        ax.set_xlim([0.,max_img])
        ax.set_ylim([0.,max_img])
        ax.set_zlim([0.,1.])
        ax.view_init(elev=25, azim=-140)
        if save:
            fig.savefig(os.path.join(path, "vuln_3d.png"),
                        bbox_inches='tight', dpi=600, format="png")
        

        # 2d contour plot vs GM1 and GM2
        fig, ax = plt.subplots()
        cp = ax.contourf(X[inds[0], inds[1]].reshape(n,n),
                   Y[inds[0], inds[1]].reshape(n,n),
                   self.Z[inds[0], inds[1]].reshape(n,n),
                   cmap=cm.coolwarm, vmin=0., vmax=1., 
                   levels=np.linspace(0,1,11))
        cbar = fig.colorbar(cp)
        cbar.set_label('$E(LR | IM_{G1}, IM_{G2})$')
        plt.scatter([1., 2.55],[2.55, 1], marker="^")
        ax.set_xlabel('{} G1 ({})'.format(imt, unit))
        ax.set_ylabel('{} G2 ({})'.format(imt, unit))
        ax.set_xlim([0.,max_img])
        ax.set_ylim([0.,max_img])
        ax.set_aspect('equal', 'box')
        if save:
            fig.savefig(os.path.join(path, "vuln_3d_contour.png"),
                        bbox_inches='tight', dpi=600, format="png")
        else:
            plt.show()

