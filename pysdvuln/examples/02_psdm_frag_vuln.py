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

import numpy as np
from pysdvuln.opensees_runner import OpenseesRunner
from pysdvuln.psdm_gg21 import PSDM_gg21
from pysdvuln.damg_state_classifier import DamgStateClassifier
from pysdvuln.damg_dep_frag_curves_noc import DamgDepFragCurvesNoC
from pysdvuln.damg_dep_frag_curves import DamgDepFragCurves
from pysdvuln.prob_collapse import ProbCollapse
from pysdvuln.damg_dep_vuln_curves import DamgDepVulnCurves
from pysdvuln.damg2loss import Damg2Loss
from pysdvuln.vuln_3d import Vuln3d
from pysdvuln.utils_pickle import load_pickle

'''
this script defines the PSDM, the collapse model (if requested), the fragility
model and the vulnerability models (given a damage-to-loss model)
look for #TODO to customize
'''

if __name__ == "__main__":
      
    #TODO decide if collapse model should be included
    consider_collapse_model = True
    
    # only useful to define the damage classifier
    capacity_curve = np.array([[0.       , 0.       ],
                               [0.0026003, 0.17158  ],
                               [0.021    , 0.343    ],
                               [0.058    , 0.292    ],
                               [0.153    , 0.295    ]])
    
    
    #%% define damage classifier, response analyses
    
    #TODO damage classifier can be defined in several ways
    # here, we define it through the capacity_curve
    opr = OpenseesRunner(capacity_curve)
    dsc = DamgStateClassifier(capacity_curve=opr.capacity_curve, cov=0.45)
    dsc.check_plots()
    
    #TODO upload response analyses file (change name accordingly)
    ims_g1, ims_g2, hysts_g1, hysts_g2, maxds_g1, maxds = load_pickle("response")
    

    #%% PSDM from Gentile and Galasso (2021)
    
    #TODO zero_tol can be changed to exclude zero Eh
    # note that "maxds" are used to get the collapsed cases, which can be also
    # done with "collapses" input
    psdm = PSDM_gg21(ims_g1, ims_g2, hysts_g1, hysts_g2, maxds_g1, dsc, 
                     maxds=maxds, zero_tol=1e-2)
    psdm.check_plots(unit="g", imt="IM")
    

    #%% fragility
    #TODO building-to-building variability sigmab2b can be changed
    
    if consider_collapse_model:
        # fragility no collapses from Gentile and Galasso (2021)
        ddfc = DamgDepFragCurvesNoC(psdm, dsc, sigmab2b=0.3, imt="IM")
        ddfc.check_plots(unit="g")
    else:
        # probability of collpse
        # note you can avoid ProbCollapse if a few collapses are observed
        # just set consider_collapse_model = False at the top
        pc = ProbCollapse(psdm)
        pc.check_plots(unit="g", imt="IM")
    
        # fragility include collapses
        ddfc = DamgDepFragCurves(psdm, dsc, sigmab2b=0.3, imt="IM")
        ddfc.check_plots(unit="g")
    
        
    #%% get fragility models and plot
    
    fig, ax = ddfc.plot_frag_all(unit="g",
                                 x_ims=np.logspace(np.log10(1e-3), np.log10(3), 1000))
    df = ddfc.get_fragilities_df()
    
    
    #%% define damage to loss model
    
    #TODO damage-to-loss models can be defined also with mean_loss, cov_loss
    d2l = Damg2Loss.default()

    
    #%% Vulnerability models and surface
    
    ddvc = DamgDepVulnCurves(ddfc, d2l)
    ddvc.check_plots(unit="g", max_img=3.)
    df = ddvc.get_vuln_curves_df()
    

    #%% Vulnerability surface
    
    v3d = Vuln3d(ddvc)
    v3d.plot_vuln_surf(unit="g", max_img=3.)
    df_3d = v3d.get_vuln_3d_df()
        