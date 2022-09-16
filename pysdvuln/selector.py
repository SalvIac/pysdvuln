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

import time
import numpy as np
from scipy.optimize import dual_annealing
import matplotlib.pyplot as plt


class Selector():
    
    def __init__(self, ims, rsns, min_im=None, max_im=None, bins=20, 
                 imt="IM (m/s2)"):
        self.ref_ims = ims
        self.bins = bins
        self.range_im = [min_im, max_im]
        if min_im is None:
            self.range_im[0] = 0.
        if max_im is None:
            self.range_im[1] = np.ceil(np.max(self.ref_ims)*1000)/1000 # np.max(self.ref_ims)
        # if rsns and imt are both None, this class is basically a plotter
        self.rsns = rsns
        if ims is not None:
            self.num_records = len(ims)
        else:
            self.num_records = None
        self.imt = imt


    def get_pair_combos(self):
        ''' all pair combination '''
        combos = list()
        for i1 in self.ref_ims:
            for i2 in self.ref_ims:
                combos.append([i1, i2])
        combos = np.array(combos)
        return combos
    

    def get_distr_im(self, numbins=50, plot=False):
        ''' actual distribution of IMs '''
        bins = np.linspace(self.range_im[0], self.range_im[1], numbins+1)
        hist, _ = np.histogram(self.ref_ims, bins=bins)
        actual_probs = hist/np.sum(hist)
        if plot:
            cents = bins[:-1] + np.diff(bins)/2
            fig, ax = plt.subplots(2,1)
            ax[0].bar(cents, actual_probs, width=0.95*np.diff(bins), edgecolor='k')
            ax[0].set_ylabel("PMF")
            # ax[0].set_yscale("log")
            ax[1].bar(cents, np.cumsum(actual_probs), width=0.95*np.diff(bins), edgecolor='k')
            ax[1].set_xlabel(self.imt)
            ax[1].set_ylabel("CDF")
            plt.show()            
        return actual_probs, bins


    def get_distr_indeces(self, numbins=50):
        ''' actual distribution of IMs '''
        _, bins = self.get_distr_im(numbins)
        indeces = list()
        for e, edge in enumerate(bins[:-1]):
            indeces.append(list())
            for i, val in enumerate(self.ref_ims):
                if edge <= val < bins[e+1]:
                    indeces[-1].append(i)
        return indeces


    def sample1d_ind(self, num=500):
        ''' sampling with with indeces '''
        ranges = np.linspace(-0.5, self.ref_ims.shape[0]-0.5, num+1)
        inds = [int(np.round(np.random.uniform(rang, ranges[i+1]))) 
                for i, rang in enumerate(ranges[:-1])]
        sampled_ims = self.ref_ims[inds]
        return sampled_ims
        
    
    def sample2d_ind(self, num=500):
        ''' 2d sampling with indeces '''
        ranges = np.linspace(-0.5, self.ref_ims.shape[0]-0.5, num+1)
        inds1 = [int(np.round(np.random.uniform(rang, ranges[i+1])))
                 for i, rang in enumerate(ranges[:-1])]
        np.random.shuffle(inds1)
        inds2 = [int(np.round(np.random.uniform(rang, ranges[i+1])))
                 for i, rang in enumerate(ranges[:-1])]
        np.random.shuffle(inds2)
        sample2d = self.get_sample2d_inds(inds1, inds2)
        return inds1, inds2, sample2d


    def sample1d_im(self, num=500, numbins=50):
        ''' sampling with with IMs '''
        from scipy.stats import qmc
        sampler = qmc.LatinHypercube(d=1)
        sample = sampler.random(n=num)
        # actual probability distribution of ims
        indeces = self.get_distr_indeces(numbins=numbins)
        actual_probs, _ = self.get_distr_im(numbins=numbins, plot=False)
        cumprobs = np.cumsum(actual_probs)
        # index
        inds1d = np.searchsorted(cumprobs, sample)
        sample1d = list()
        for row in inds1d:
            # sample CMF
            if len(indeces[row[0]]) == 0:
                raise Exception("error")
            ind = np.random.choice(indeces[row[0]])
            sample1d.append(self.ref_ims[ind])
        sample1d = np.array(sample1d)
        return sample1d
        

    def sample2d_im(self, num=500, numbins=50):
        ''' 2d Latin Hypercube with IMs '''
        from scipy.stats import qmc
        sampler = qmc.LatinHypercube(d=2)
        sample = sampler.random(n=num)
        # qmc.discrepancy(sample)
        # actual probability distribution of ims
        indeces = self.get_distr_indeces(numbins=numbins)
        actual_probs, _ = self.get_distr_im(numbins=numbins, plot=False)
        cumprobs = np.cumsum(actual_probs)
        # index
        inds2d = np.searchsorted(cumprobs, sample)
        sample2d = list()
        for row in inds2d:
            # sample CMF twice
            temp = list()
            for i in range(0,2):
                if len(indeces[row[i]]) == 0:
                    raise Exception("error")
                ind = np.random.choice(indeces[row[i]])
                temp.append(self.ref_ims[ind])
            sample2d.append(temp)
        sample2d = np.array(sample2d)
        return sample2d

    
    @classmethod
    def decode_inds(cls, all_inds, n, astype=int):
        inds_gm1 = np.array(all_inds[:n]).flatten().astype(astype)
        inds_gm2 = np.array(all_inds[n:]).flatten().astype(astype)
        return inds_gm1, inds_gm2


    def get_sample2d_inds(self, inds_gm1, inds_gm2, sf_gm1=1., sf_gm2=1.):
        return np.vstack([sf_gm1*self.ref_ims[inds_gm1],
                          sf_gm2*self.ref_ims[inds_gm2]]).T


    def get_siteclass2d_inds(self, inds_gm1, inds_gm2):
        # site_class
        return np.vstack([self.siteclass2d[inds_gm1],
                          self.siteclass2d[inds_gm2]]).T


    @classmethod
    def evaluate_sample2d(cls, sample2d, range_im, bins):
        hist, _, _= np.histogram2d(sample2d[:,0], sample2d[:,1],
                                   bins=bins, range=[range_im, range_im])
        return cls.get_mse(hist)


    @classmethod
    def evaluate_siteclass2d(cls, siteclass2d):
        hist, _, _= np.histogram2d(siteclass2d[:,0], siteclass2d[:,1],
                                   bins=[-0.5,0.5,1.5,2.5,3.5,4.5])
        return cls.get_mse(hist)
    

    @classmethod
    def get_mse(cls, hist):
        # mean squared error
        matr = hist - np.mean(hist)
        matr[matr > np.mean(hist)] *= 10
        mse = np.square(matr).mean()
        if mse == 0:
            mse = 1e-3
        # num_nonzero = np.count_nonzero(hist)
        # tot_bins = hist.shape[0]*hist.shape[1]
        if False: #num_nonzero != tot_bins: # False: # as many bins as possible with a pair
            outliers1 = 1. + np.sum(hist - np.mean(hist) > 1.)
            outliers2 = 1. + 10*np.max(hist - np.mean(hist))
            return np.log((tot_bins - num_nonzero)*outliers1*outliers2*mse)
        else:
            return np.log(mse)

    
    def fitness_pair_unscaled(self, x):
        '''
        x is a 1d numpy array (n: num ground-motion pairs to be selected):
        [index GM1 * n, index GM2 * n]
        '''
        raise Exception("recheck and update this function before using it")
        n = int(len(x)/2)
        inds1, inds2 = self.decode_inds(x, n)
        inds = np.vstack([inds1, inds2]).T
        if np.unique(inds, axis=0).shape[0] != inds.shape[0]:
            return n + (inds.shape[0] - np.unique(inds, axis=0).shape[0])*n
        sample2d = self.get_sample2d_inds(inds1, inds2)
        return self.evaluate_sample2d(sample2d, self.range_im, self.bins)
        
    
    # old version with ids instead of rsn
    # def fitness_pair(self, x):
    #     '''
    #     x is a 1d numpy array:
    #     [index GM1 * n, index GM2 * n, scaling GM1 * n, scaling GM2 * n]
    #     '''
    #     raise Exception("recheck and update this function before using it")
    #     n = int(len(x)/4)
    #     inds1, inds2 = self.decode_inds(x[:2*n], n)
    #     inds = np.vstack([inds1, inds2]).T
    #     # check none of the "pairs" is composed by the same index for gm1 and gm2
    #     constr1 = np.sum(inds[:,0] == inds[:,1])
    #     # check all "pairs" of (ind1,ind2) are unique
    #     constr2 = inds.shape[0] - np.unique(inds, axis=0).shape[0]
    #     # check that at least 80% of inds1 are unique
    #     constr3 = int(inds1.shape[0]*0.7) - np.unique(inds1).shape[0]
    #     # check that at least 80% of inds2 are unique
    #     constr4 = int(inds2.shape[0]*0.7) - np.unique(inds2).shape[0]
    #     mult = 1.
    #     message = "ok"
    #     if constr1 > 0:
    #         message = "constr1"
    #         mult = 1+n*constr1
    #     elif constr2 > 0:
    #         message = "constr2"
    #         mult = 1+n*constr2
    #     elif constr3 > 0:
    #         message = "constr3"
    #         mult = 1+n*constr3
    #     elif constr4 > 0:
    #         message = "constr4"
    #         mult = 1+n*constr4
    #     sf1, sf2 = self.decode_inds(x[2*n:], n, float)
    #     sample2d = self.get_sample2d_inds(inds1, inds2, sf1, sf2)
    #     # check all ground motions are within the range
    #     if np.any(sample2d > self.max_im):
    #         message = "constr5"
    #         mult = 1+1e2*n*(np.sum(sample2d > self.max_im))
    #     mse = self.evaluate_sample2d(sample2d, self.max_im) + np.log(mult)
    #     print(message, mse)
    #     return mse


    def fitness_pair(self, x, return_message=False):
        '''
        x is a 1d numpy array:
        [index GM1 * n, index GM2 * n, scaling GM1 * n, scaling GM2 * n]
        '''
        n = int(len(x)/4)
        inds1, inds2 = self.decode_inds(x[:2*n], n)
        rsn1 = self.rsns[inds1]
        rsn2 = self.rsns[inds2]
        rsn = np.vstack([rsn1, rsn2]).T
        # check "pairs" must not contain the same rsn for gm1 and gm2
        constr1 = np.sum(rsn[:,0] == rsn[:,1])
        # check all "pairs" of (rsn1,rsn2) must be unique
        constr2 = rsn.shape[0] - np.unique(rsn, axis=0).shape[0]
        # check that at least 70% of rsn for gm1 must be unique
        constr3 = int(rsn1.shape[0]*0.7) - np.unique(rsn1).shape[0]
        # check that at least 70% of rsn for gm2 must be unique
        constr4 = int(rsn2.shape[0]*0.7) - np.unique(rsn2).shape[0]
        mult = 1.
        if return_message: message = ""
        if constr1 > 0:
            if return_message: message += "constr1 "
            mult = 1. + mult*n*constr1
        if constr2 > 0:
            if return_message: message += "constr2 "
            mult = 1. + mult*n*constr2
        if constr3 > 0:
            if return_message: message += "constr3 "
            mult = 1. + mult*n*constr3
        if constr4 > 0:
            if return_message: message += "constr4 "
            mult = 1. + mult*n*constr4
        sf1, sf2 = self.decode_inds(x[2*n:], n, float)
        sample2d = self.get_sample2d_inds(inds1, inds2, sf1, sf2)
        # check all ground motions are within the range
        if np.any(sample2d > self.range_im[1]):
            if return_message: message += "constr5 "
            mult = 1. + mult*n*(np.sum(sample2d > self.range_im[1]))
        if np.any(sample2d < self.range_im[0]):
            if return_message: message += "constr6 "
            mult = 1. + mult*n*(np.sum(sample2d < self.range_im[0]))
        logmse = self.evaluate_sample2d(sample2d, self.range_im, self.bins) + \
              np.log(mult) # this is to penalize logmse in case of constraints
        if return_message:
            return logmse, message
        else:
            return logmse


    # attempt at multi objective optimization
    # def fitness_pair_vs30(self, x):
    #     '''
    #     x is a 1d numpy array:
    #     [index GM1 * n, index GM2 * n, scaling GM1 * n, scaling GM2 * n]
    #     '''
    #     n = int(len(x)/4)
    #     inds1, inds2 = self.decode_inds(x[:2*n], n)
    #     inds = np.vstack([inds1, inds2]).T
    #     # unique pairs
    #     if np.unique(inds, axis=0).shape[0] != inds.shape[0]:
    #         return n + (inds.shape[0] - np.unique(inds, axis=0).shape[0])*n
    #     sf1, sf2 = self.decode_inds(x[2*n:], n, float)
    #     siteclass2d = self.get_siteclass2d_inds(inds1, inds2)
    #     sample2d = self.get_sample2d_inds(inds1, inds2, sf1, sf2)
    #     mse = self.evaluate_sample2d(sample2d, self.max_im)
    #     mse2 = self.evaluate_siteclass2d(siteclass2d)
    #     print(mse+mse2)
    #     return mse+mse2

        
    def simul_ann_unscaled(self, n, **kwargs):
        lw = [0] * 2*n
        up = [self.num_records-1e-12] * 2*n
        bounds = list(zip(lw, up))
        ret = dual_annealing(self.fitness_pair_unscaled, bounds=bounds, **kwargs)
        inds1, inds2 = self.decode_inds(ret.x, n)
        return inds1, inds2
    
        
    def simul_ann(self, n, minscale=0.5, maxscale=2., **kwargs):
        lw = [0] * 2*n + [minscale] * 2*n
        up = [self.num_records-1e-12] * 2*n + [maxscale] * 2*n 
        x0 = (np.random.choice(np.arange(self.num_records), size=2*n) + \
              np.random.uniform(size=2*n)).tolist() + \
             [0.9*maxscale] * 2*n 
        bounds = list(zip(lw, up))
        callback = Callback()
        t = time.time()
        ret = dual_annealing(self.fitness_pair, bounds=bounds, x0=x0, 
                             callback=callback, **kwargs)
                             # defaults are: 
                             # maxiter=1000, initial_temp=5230.0, 
                             # restart_temp_ratio=2e-05, visit=2.62, 
                             # accept=-5.0, maxfun=1e7, no_local_search=False
        logmse, message = self.fitness_pair(ret.x, return_message=True)
        if message == "":
            print("Constraints respected -",
                  "final logmse", logmse, "-",
                  "time", time.time()-t, "-")
        else:
            print("Constraints NOT respected:", message, "-",
                  "final logmse", logmse, "-",
                  "time", time.time()-t, "-")
        inds1, inds2 = self.decode_inds(ret.x[:2*n], n)
        sf1, sf2 = self.decode_inds(ret.x[2*n:], n, float)
        sample2d = self.get_sample2d_inds(inds1, inds2, sf1, sf2)
        # # modify
        # while np.where(sample2d[:,1] > self.range_im[0])[0].shape[0] != 0:
        #     ret.x[3*n+np.where(sample2d[:,1] > self.range_im[0])[0]] -= 0.1
        #     inds1, inds2 = self.decode_inds(ret.x[:2*n], n)
        #     sf1, sf2 = self.decode_inds(ret.x[2*n:], n, float)
        #     sample2d = self.get_sample2d_inds(inds1, inds2, sf1, sf2)        
        # while np.where(sample2d[:,0] > self.range_im[0])[0].shape[0] != 0:
        #     ret.x[2*n+np.where(sample2d[:,0] > self.range_im[0])[0]] -= 0.1
        #     inds1, inds2 = self.decode_inds(ret.x[:2*n], n)
        #     sf1, sf2 = self.decode_inds(ret.x[2*n:], n, float)
        #     sample2d = self.get_sample2d_inds(inds1, inds2, sf1, sf2) 
        return inds1, inds2, sf1, sf2, sample2d


    def scatter(self, array1, array2, scale=1., **kwargs):
        fig, ax = plt.subplots(figsize=(6,6))
        ax.scatter(array1*scale, array2*scale, **kwargs)#s=20, linewidth=0.1, edgecolors='k', color='gray')
        ax.set_xlabel("GM1 "+self.imt)
        ax.set_ylabel("GM2 "+self.imt)
        plt.show()
        return fig, ax
        
        
    def hist2d(self, array2d, scale=1., centered=False, mask_zero=True):
        hist, xedges, yedges = np.histogram2d(array2d[:,0]*scale,
                                              array2d[:,1]*scale,
                                              bins=self.bins, 
                                              range=[[r*scale for r in self.range_im],
                                                     [r*scale for r in self.range_im]])
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        if centered:
            xpos, ypos = np.meshgrid(xedges[:-1] + np.min(np.diff(xedges))/2,
                                     yedges[:-1] + np.min(np.diff(yedges))/2, 
                                     indexing="ij")
        else:
            xpos, ypos = np.meshgrid(xedges[:-1], xedges[:-1], indexing="ij")
        xpos = xpos.ravel()
        ypos = ypos.ravel()
        zpos = 0
        bar_width = np.min(np.diff(xedges))*0.8
        dx = dy = bar_width*np.ones_like(zpos)
        dz = hist.ravel()
        if mask_zero:
            mask_dz = dz == 0
        else:
            mask_dz = np.ones_like(xpos, dtype=bool)
        ax.bar3d(xpos[~mask_dz], ypos[~mask_dz], zpos, dx, dy, dz[~mask_dz],
                 zsort='average')
        ax.set_xlabel("GM1 "+self.imt)
        ax.set_ylabel("GM2 "+self.imt)
        plt.show()
        return fig, ax

    
    def __repr__(self):
        return self.__str__()
        
    def __str__(self):
        return "<{}, {} bins, range im {}, num records {}>".format(
                self.__class__.__name__, self.bins, self.range_im, self.num_records)


    # siteclass2d = rc.get_siteclass2d_inds(inds1, inds2)
    # hist2, xedges2, yedges2 = np.histogram2d(siteclass2d[:,0], siteclass2d[:,1],
    #                                bins=[-0.5,0.5,1.5,2.5,3.5,4.5])


    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # xpos, ypos = np.meshgrid(xedges2[:-1], xedges2[:-1], indexing="ij")
    # xpos = xpos.ravel()
    # ypos = ypos.ravel()
    # zpos = 0
    # dx = dy = np.ones_like(zpos)
    # dz = hist2.ravel()
    # ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
    # ax.set_xlabel("Site class GM1")
    # ax.set_ylabel("Site class GM2")
    # ax.set_xticks(range(5))
    # ax.set_xticklabels(["A","B","C","D","E"])
    # ax.set_yticks(range(5))
    # ax.set_yticklabels(["A","B","C","D","E"])
    # plt.show()
    
    
    # mags1 = rc.get_array("magnitude")[inds1]
    # dists1 = rc.get_array("epicentral_distance")[inds1]
    # mags2 = rc.get_array("magnitude")[inds2]
    # dists2 = rc.get_array("epicentral_distance")[inds2]

    
    # hist2, xedges2, yedges2 = np.histogram2d(mags1, mags2)
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # xpos, ypos = np.meshgrid(xedges2[:-1], xedges2[:-1], indexing="ij")
    # xpos = xpos.ravel()
    # ypos = ypos.ravel()
    # zpos = 0
    # dx = dy = 0.2*np.ones_like(zpos)
    # dz = hist2.ravel()
    # ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
    # ax.set_xlabel("mag GM1")
    # ax.set_ylabel("mag GM2")
    # plt.show()    
    
    
    # hist2, xedges2, yedges2 = np.histogram2d(dists1, dists2)
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # xpos, ypos = np.meshgrid(xedges2[:-1], xedges2[:-1], indexing="ij")
    # xpos = xpos.ravel()
    # ypos = ypos.ravel()
    # zpos = 0
    # dx = dy = 10*np.ones_like(zpos)
    # dz = hist2.ravel()
    # ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
    # ax.set_xlabel("dist GM1")
    # ax.set_ylabel("dist GM2")
    # plt.show()
    


class Callback():
    
    def __init__(self):
        self.iter = 0
        
    def __call__(self, x, f, context):
        '''
        A callback function with signature callback(x, f, context), which will
        be called for all minima found. x and f are the coordinates and 
        function value of the latest minimum found.
        If the callback implementation returns True, the algorithm will stop.
        '''
        print(self.iter, f)
        self.iter += 1
        #TODO I can add stopping conditions here (return True for stop)
        
