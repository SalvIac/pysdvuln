# -*- coding: utf-8 -*-
"""
@author: Salvatore Iacoletti
"""

from copy import deepcopy
import pandas as pd
from tqdm import tqdm
from myutils.utils_pickle import save_pickle, load_pickle
from pysdvuln.opensees_runner import OpenseesRunner
from pysdvuln.damg_state_classifier import DamgStateClassifier
from pysdvuln.psdm_gg21 import PSDM_gg21
from pysdvuln.damg_dep_frag_curves_noc import DamgDepFragCurvesNoC
from pysdvuln.damg_dep_frag_curves import DamgDepFragCurves


if __name__ == "__main__":
    
    capacity_curve = pd.read_csv("capacity_curve.csv").to_numpy()
    dsc = DamgStateClassifier(capacity_curve=capacity_curve)#, cov=0.45)
    dsc.COLLAPSE_FACTOR = 1.
    

    #%% run all selected ground motions with opensees    

    # opr = OpenseesRunner(capacity_curve)
    # rec_pairs = load_pickle(r"C:\Users\Salvatore\Dropbox\SalvIac\pysdvuln\pysdvuln\tests\test_fragility\selection_T_0_513\nga_800_rec_pairs")
    # res = list()
    # for i, rec in tqdm(enumerate(rec_pairs)):
    #     _ = opr.run(rec["rp"].get_time_gmr(unit="m/s2"), mat="Steel01")
    #     res.append( deepcopy(rec) )
    #     res[-1]["opr"] = deepcopy(opr)
    # save_pickle(res, "nltha")
    res = load_pickle("nltha")
    
    
    #%% PSDM and fragility curves from Gentile and Galasso (2021)

    psdm = PSDM_gg21.from_analyses(res, dsc, zero_tol=1e-1)
    # psdm.check_plots()
    
    ddfc = DamgDepFragCurvesNoC(psdm, dsc)
    ax = ddfc.plot_frag_all(unit="g", imt="avgSA")
    ax.set_xlim(0,3)
    df = ddfc.get_frags_params_df(unit="g")
    
    
    #%% probability of collapse
    
    from pysdvuln.prob_collapse import ProbCollapse
    pc = ProbCollapse(psdm)
    pc.check_plots()


    #%% fragility with probability of collapse
    
    ddfc1 = DamgDepFragCurves(psdm, dsc)
    ax = ddfc1.plot_frag_all(unit="g", imt="avgSA")
    ax.set_xlim(0,3)
    df1 = ddfc1.get_frags_params_df(unit="g")

