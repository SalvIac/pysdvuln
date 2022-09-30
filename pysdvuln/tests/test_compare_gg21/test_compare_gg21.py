# -*- coding: utf-8 -*-
"""
@author: Salvatore Iacoletti
"""

import numpy as np
import pandas as pd
from pysdvuln.damg_state_classifier import DamgStateClassifier
from pysdvuln.psdm_gg21 import PSDM_gg21
from pysdvuln.damg_dep_frag_models_noc import DamgDepFragModelsNoC
from scipy.io import loadmat



if __name__ == "__main__":

    build_class = 2

    # load data from Roberto
    rob = loadmat("gentileAftershockEhEESD.mat")
    print("build_class", rob["CASEnames"][0][build_class][0])
    
    maxds_g1 = rob["driftMaxMS"][0][build_class].flatten()
    ims_g1 = rob["avgSAms"][:,build_class]
    hysts_g1 = rob["EhMSnorm"][0][build_class].flatten()
    maxds_g2 = rob["driftMaxAS"][0][build_class].flatten()
    ims_g2 = rob["avgSAas"][:,build_class]
    hysts_g2 = rob["EhASnorm"][0][build_class].flatten()
    
    collapses = rob["collapsed"][0][build_class].flatten() != 0
    
    # outliers (asked rob for these filters)
    collapses[hysts_g1 <= 0.] = True
    collapses[hysts_g2 <= 0.] = True
    if build_class == 7:
        collapses[hysts_g1 >= 4.5] = True
        collapses[hysts_g2 >= 4.5] = True
    
    
    #%% damage state classifier
    
    if build_class == 2:
        thresholds = [0.03, 0.19, 1.32, 1.76]

    thresholds = rob["DSthresholdsDrift"][build_class,:4]
    dsc = DamgStateClassifier(thresholds=thresholds)


    #%% PSDM and fragility curves from Gentile and Galasso (2021)

    psdm = PSDM_gg21(ims_g1, ims_g2, hysts_g1, hysts_g2, maxds_g1,
                     dsc, collapses=collapses, zero_tol=1e-6, 
                     mode="original") # modified

    psdm.check_plots()
    df_psdm = psdm.get_params_df()


    #%% check psdm results

    if build_class == 2:
        # results ms_bare_4
        data = {"a":  2.2,
                "b":  1.37,
                "c0": 6.66,
                "d":  1.68,
                "m":  0.15}
        psdm_check = pd.DataFrame(data, index=["check"]).T
    elif build_class == 7:
        # results bs_infilled_8
        data = {"a":  0.59,
                "b":  1.29,
                "c0": 2.71,
                "d":  1.67,
                "m":  0.18}
        psdm_check = pd.DataFrame(data, index=["check"]).T
    print(pd.concat([psdm_check, df_psdm], axis=1))
    

    #%% fragility curves 
    
    # repeat the PSDM with avgSA in m/s2
    psdm = PSDM_gg21(ims_g1*9.81, ims_g2*9.81, hysts_g1, hysts_g2, maxds_g1,
                     dsc, collapses=collapses, zero_tol=1e-6)
    
    # no collapses
    ddfc = DamgDepFragModelsNoC(psdm, dsc)
    ddfc.check_plots(unit="g")
    df_frag = ddfc.get_frags_params_df(unit="g")
    
    
    #%% normality test

    import matplotlib.pyplot as plt
    from scipy import stats
    from scipy import interpolate
    from scipy.integrate import simps
    

    # # histogram residuals
    mu, std = stats.norm.fit(ddfc.residuals_log) # Fit a normal distribution to the data
    
    # histogram residuals
    mu, std = stats.norm.fit(ddfc.residuals_log) # Fit a normal distribution to the data
    _, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    h, vals = np.histogram(ddfc.residuals_log, bins=100, density=True)
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

    
    k2, p = stats.normaltest(ddfc.residuals_log)
    alpha = 1e-3
    print("p = {:g}".format(p))
    if p < alpha:  # null hypothesis: x comes from a normal distribution
        print("The null hypothesis can be rejected")
    else:
        print("The null hypothesis cannot be rejected")
    
    
    res = stats.kstest(ddfc.residuals_log, 'norm')
    print(res.pvalue)
    if res.pvalue < 0.05:  # null hypothesis: x comes from a normal distribution
        print("The null hypothesis can be rejected")
    else:
        print("The null hypothesis cannot be rejected")
    
    
    
    pr = np.linspace(0.001,0.999,10000)
    yp = stats.norm.ppf(pr, mu, std)
    yp1 = f(pr)
    print("EMD", np.sqrt(simps(pr, (yp1-yp)**2)),
          np.sqrt(simps(pr, (yp1-yp)**2)) / np.mean(ddfc.residuals_log), 
          np.mean(ddfc.residuals_log))
    



    # histogram residuals
    mu, std = stats.norm.fit(ddfc.residuals) # Fit a normal distribution to the data
    _, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    h, vals = np.histogram(ddfc.residuals, bins=100, density=True)
    ax1.bar(vals[:-1]+np.diff(vals)/2, h, width=0.9*min(np.diff(vals)))
    xmin, xmax = ax1.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    ax1.plot(x, stats.norm.pdf(x, mu, std), 'k', linewidth=2)
    ax1.set_ylabel("PDF")
    ax2.plot(x, stats.norm.cdf(x, mu, std), 'k', linewidth=2)
    ax2.plot(vals[:-1]+np.diff(vals)/2, np.cumsum(h)*np.diff(vals))
    ax2.set_xlabel("lin-space residuals")
    ax2.set_ylabel("CDF")
    

    k2, p = stats.normaltest(ddfc.residuals)
    alpha = 1e-3
    print("p = {:g}".format(p))
    if p < alpha:  # null hypothesis: x comes from a normal distribution
        print("The null hypothesis can be rejected")
    else:
        print("The null hypothesis cannot be rejected")
    
    
    res = stats.kstest(ddfc.residuals, 'norm')
    print(res.pvalue)
    if res.pvalue < 0.05:  # null hypothesis: x comes from a normal distribution
        print("The null hypothesis can be rejected")
    else:
        print("The null hypothesis cannot be rejected")


    pr = np.linspace(0.001,0.999,10000)
    yp = stats.norm.ppf(pr, mu, std)
    yp1 = f(pr)
    print("EMD", np.sqrt(simps(pr, (yp1-yp)**2)),
          np.sqrt(simps(pr, (yp1-yp)**2)) / np.mean(ddfc.residuals), 
          np.mean(ddfc.residuals))

    
    
    
    #%% check fragility results

    if build_class == 2:
        # results ms_bare_4
        data = {"DS1":      0.03,
                "DS2":      0.14,
                "DS3":      0.33,
                "DS4":      0.42,
                "DS2|DS1":  0.13,
                "DS3|DS1":  0.33,
                "DS4|DS1":  0.42,
                "DS3|DS2":  0.29,
                "DS4|DS2":  0.39,
                "DS4|DS3":  0.23,
                "beta":     0.35}
        frag_check = pd.DataFrame(data, index=["check"]).T
    elif build_class == 7:
        # results bs_infilled_8
        data = {"DS1":      0.02,
                "DS2":      0.10,
                "DS3":      0.98,
                "DS4":      1.25,
                "DS2|DS1":  0.10,
                "DS3|DS1":  0.98,
                "DS4|DS1":  1.26,
                "DS3|DS2":  0.98,
                "DS4|DS2":  1.27,
                "DS4|DS3":  1.07,
                "beta":     0.30}
        frag_check = pd.DataFrame(data, index=["check"]).T
    print(pd.concat([frag_check, df_frag], axis=1))

    