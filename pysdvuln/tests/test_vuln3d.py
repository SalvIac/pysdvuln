# -*- coding: utf-8 -*-
"""
@author: Salvatore Iacoletti
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from myutils.utils_pickle import save_pickle, load_pickle
from pysdvuln.opensees_runner import OpenseesRunner
from pysdvuln.psdm_gg21 import PSDM_gg21
from pysdvuln.damg_state_classifier import DamgStateClassifier
from pysdvuln.damg_dep_frag_models_noc import DamgDepFragModelsNoC
from pysdvuln.damg_dep_frag_models import DamgDepFragModels
from pysdvuln.damg_dep_vuln_models import DamgDepVulnModels
from pysdvuln.base_sdof import BaseSDOF
bsd = BaseSDOF()
from pysdvuln.prob_collapse import ProbCollapse
from pysdvuln.damg2loss import Damg2Loss
from pysdvuln.vuln_3d import Vuln3d
from pysdvuln.groupers import (PeriodGrouper, YieldCapGrouper,
                                      YieldCapGrouper2, duct2ry)
from myutils.search_file_in_folder import (search_filenames_in_folder,
                                           search_filepaths_in_folder)


if __name__ == "__main__":
    
    classes =  ['CR_LFM-DUL_H2', 'CR_LFM-DUL_H5', 'CR_LFM-DUM_H2', 'CR_LFM-DUM_H5',
           'MR_LWAL-DUL_H2', 'MR_LWAL-DUL_H5', 'MR_LWAL-DUM_H2',
           'MR_LWAL-DUM_H5', 'MUR_LWAL-DNO_H2', 'MUR_LWAL-DNO_H5',
           'W_LFM-DUL_H2', 'W_LFM-DUL_H5', 'W_LFM-DUM_H2', 'W_LFM-DUM_H5'] #["CR_LFINF-DUL_H4"]
    
    data_folder = r"C:\Users\Salvatore\Dropbox\SalvIac\pyphd\results\20220220_run_database\adhoc\ss"
    
    curve = classes[0]
    path = os.path.join(data_folder, curve)

    # Fragility curves
    ddfc = load_pickle(path+"/ddfc")
    # ddfc.plot_frag_all(max_img=3.)
    
    # Vulnerability curves
    ddvc = load_pickle(path+"/ddvc")
    # ddvc.check_plots(unit="g", max_img=3.)
    # df = pd.read_csv(path+"/vulns.csv")

    # 3d surface
    v3d = Vuln3d.from_ddvc(ddvc)
    # v3d.plot_vuln_surf(unit="g", max_img=3.)
    
    
    #%% other way


    def prob_dss(poes):
        """
        it returns the probability of having a specific DS for the given
        Intensity Measure Levels (IMLs). rows correspond to the ds, columns
        correspond to the imls
        """
        poes = np.insert(poes, 0, 1., axis=0)
        poes[poes<1e-3] = 0. # adjust to avoid numerical errors
        pr = poes[:-1,:] - poes[1:,:]
        pr = np.insert(pr, pr.shape[0], poes[-1,:], axis=0)
        return pr


    inds_closest = list()
    closest = list()
    for i in [0., 0.1, 0.2, 0.3]: # , 0.5,1.,1.5,2.
        inds_closest.append(np.abs(ddfc.x_ims_g-i).argmin())
        closest.append( ddfc.x_ims_g[inds_closest[-1]] )
    inds_closest = np.array(inds_closest)
        
    
    im_g1 = ddfc.x_ims_g # np.array(closest) # ddfc.x_ims_g # np.array([0.5, 1., 1.5, 2.]) #  # 
    im_g2 = ddfc.x_ims_g # np.array(closest) # ddfc.x_ims_g # np.array(closest) # 

    #TODO hardcoded 4 DSs
    num_sims = 10000
    num_dss = 4
    # vals2 = np.zeros((len(im_g1), len(im_g2), num_dss*num_sims)) # num_dss)) #num_dss*num_sims))
    eps = np.linspace(0,1,num_sims+1) # np.random.rand(num_sims) # 
    eps = eps[:-1]+np.diff(eps)/2
    
    # ds0 here is the state of the structure before GM1
    for ds0 in range(0,num_dss):
        
        # eps = np.random.rand(num_sims)
        
        # get transitioning probabilities
        frags = list()
        for ds1 in range(ds0+1, 5): 
             frags.append( ddfc.get_fragility(im_g1*9.81, ds1, ds0) )
        poes = np.array(frags)
        pr = prob_dss(poes)
        
        # plot
        cols = [[0.5,0.5,0.5], "b", "m", "g", "r"]
        ax = ddfc.plot_frag_ds(ds1_plot=ds0, max_img=3.)
        # plot bar representing transitioning probabilities
        for c in range(pr.shape[1]):
            for r in range(1,len(pr[:,c])):
                ax.bar(im_g1[c], pr[r,c], 0.05, bottom=np.sum(pr[r+1:,c]), 
                        color=cols[r+ds0])
            ax.bar(im_g1[c], pr[0,c], 0.05, bottom=np.sum(pr[1:,c]),
                    color=cols[0+ds0])
        plt.show()
        stop
        # get DSs for each eps (this is DS1)
        # rows are the reached DSs (DS1)
        # columns are the IM levels
        dss = list()
        for c in range(pr.shape[1]):
            dss.append( ds0+np.searchsorted(np.cumsum(pr[:,c],axis=0).flatten(), eps) )
        
        dss = np.array(dss).T
        dss = np.clip(dss, 0, 4)
        vals = np.zeros((len(im_g1), len(im_g2), len(eps)))
        for r in range(dss.shape[0]):
            for c in range(dss.shape[1]):
                if dss[r,c] == 4:
                    vals[c, :, r] = 1.
                else:
                    vals[c, :, r] = ddvc.interpolate(im_g2, dss[r,c])
        vals2[:,:,ds0*len(eps):(ds0+1)*len(eps)] = vals #np.mean(vals, axis=2) # ds0*len(eps):(ds0+1)*len(eps)
        
        # plot 3d surface
        ax = v3d.plot_vuln_surf(unit="g", max_img=3.)
        for r in range(im_g1.shape[0]):
            for c in range(im_g2.shape[0]):
                # ax.scatter(im_g1[r], im_g2[c], vals[r, c, :], color=[0.5,0.5,0.5])
                ax.scatter(im_g1[r], im_g2[c], np.mean(vals[r, c, :]), color="r")
        
        # X, Y = np.meshgrid(inds_closest, inds_closest)
        # diff = v3d.Z[X, Y] / np.mean(vals, axis=2) - 1.
        diff = v3d.Z / np.mean(vals, axis=2) - 1.
        ax = Vuln3d(im_g1, im_g2, diff).plot_vuln_surf(unit="g", max_img=3.)
        ax.relim()      # make sure all the data fits
        ax.autoscale()  # auto-scale
        ax.set_zlabel("Ratio")
        stop
    
    # # plot old 3d surface 
    # ax = v3d.plot_vuln_surf(unit="g", max_img=3.)
    # for r in range(im_g1.shape[0]):
    #     for c in range(im_g2.shape[0]):
    #         # ax.scatter(im_g1[r], im_g2[c], vals[r, c, :], color=[0.5,0.5,0.5])
    #         ax.scatter(im_g1[r], im_g2[c], np.mean(vals2[r, c, :]), color="r")
    
    # plot final surface
    # v3d_test = Vuln3d(im_g1, im_g2, np.mean(vals, axis=2))
    # v3d_test.plot_vuln_surf(unit="g", max_img=3.)
    


