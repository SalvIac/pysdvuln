# -*- coding: utf-8 -*-
"""
@author: Salvatore Iacoletti
"""

from copy import deepcopy
import pandas as pd
from tqdm import tqdm
from myutils.utils_pickle import save_pickle, load_pickle
from pysdvuln.opensees_runner import OpenseesRunner
from pysdvuln.damage_state_classif import DamageStateClassif
from pysdvuln.psdm_gg21 import PSDM_gg21
from pysdvuln.damg_dep_frag_curves_noc import DamgDepFragCurvesNoC


if __name__ == "__main__":
    
    capacity_curve = pd.read_csv("capacity_curve.csv").to_numpy()
    opr = OpenseesRunner(capacity_curve)
    dsc = DamageStateClassif(capacity_curve=opr.capacity_curve)


    #%% run all selected ground motions with opensees    
    
    # rec_pairs = load_pickle(r"C:\Users\Salvatore\Dropbox\SalvIac\pysdvuln\pysdvuln\tests\test_fragility\selection_T_0_229\nga_800_rec_pairs")
    # res = list()
    # for i, rec in tqdm(enumerate(rec_pairs)):
    #     _ = opr.run(rec["rp"].get_time_gmr(unit="m/s2"))
    #     res.append( deepcopy(rec) )
    #     res[-1]["opr"] = deepcopy(opr)
    # save_pickle(res, "nltha")
    res = load_pickle("nltha")
    
    
    #%% PSDM and fragility curves from Gentile and Galasso (2021)

    psdm = PSDM_gg21(res, dsc)
    psdm.check_plots()
    
    ddfc = DamgDepFragCurvesNoC(psdm, dsc)
    ax = ddfc.plot_frag_all(unit="g", imt="avgSA")
    ax.set_xlim(0,3)
    df = ddfc.get_frags_params_df(unit="g").T
    