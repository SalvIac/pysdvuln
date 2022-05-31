# -*- coding: utf-8 -*-
"""
@author: Salvatore Iacoletti
"""

import numpy as np
from scipy.stats import expon
from pysdvuln.selector import Selector


if __name__ == "__main__":
   
    # random ims and rsns
    ims = expon.rvs(loc=0.05, scale=3., size=4000)
    rsns = np.array([i+1 for i in range(2000) for _ in range(2)])
    

    #%% simulating annealing selection

    sel = Selector(ims, rsns, imt="IM")
    n = 400


    #%% simple selection sorting PGA
    
    r0 = ims #rc.get_sas(0.)
    inds = np.argsort(r0)[-800:]
    np.random.shuffle(inds)
    inds1 = inds[:400]
    inds2 = inds[400:800]
    
    sample2d = sel.get_sample2d_inds(inds1, inds2)
    sel.scatter(sample2d[:,0], sample2d[:,1], s=10, lw=0.5, edgecolor="k")
    sel.hist2d(sample2d, bar_width=0.5)

    
    #%% various kind of sampling 1d and 2d
    
    combos = sel.get_pair_combos()
    sel.scatter(combos[:,0], combos[:,1], s=10, lw=0.5, edgecolor="k")
    sel.hist2d(combos, bar_width=0.5)

    sel.get_distr_im(numbins=50, plot=True)
    
    sel.sample1d_ind(num=n)

    _, _, sample2d = sel.sample2d_ind(num=n)
    sel.scatter(sample2d[:,0], sample2d[:,1], s=10, lw=0.5, edgecolor="k")
    sel.hist2d(sample2d, bar_width=0.5)
    
    sel.sample1d_im(num=n, numbins=50)

    sample2d = sel.sample2d_im(num=n, numbins=50)
    sel.scatter(sample2d[:,0], sample2d[:,1], s=10, lw=0.5, edgecolor="k")
    sel.hist2d(sample2d, bar_width=0.5)

    
    #%% simulating annealing selection
    
    inds1, inds2, sf1, sf2, sample2d = sel.simul_ann(n, maxiter=10)
    
    # plots selection
    sel.scatter(sample2d[:,0], sample2d[:,1], s=10, lw=0.5, edgecolor="k")
    sel.hist2d(sample2d, bar_width=0.5)
    
