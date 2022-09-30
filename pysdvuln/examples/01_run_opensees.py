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
from tqdm import tqdm
from pysdvuln.utils_pickle import save_pickle, load_pickle
from pysdvuln.opensees_runner import OpenseesRunner
from pysdvuln.record_pair import RecordPair
from pysdvuln.psdm_gg21 import PSDM_gg21
from pysdvuln.base_sdof import BaseSDOF
bsd = BaseSDOF()

'''
run opensees with SDOF and selected ground motions
output is a tuple (ims_g1, ims_g2, hysts_g1, hysts_g2, maxds_g1, maxds)
look for #TODO to customize
'''

if __name__ == "__main__":
    
    #TODO upload database file (change filename accordingly)
    rc = load_pickle("database_gms")
    
    #TODO define capacity curve (Sd-Sa)
    opr = OpenseesRunner(np.array([[0.       , 0.       ],
                                   [0.0026003, 0.17158  ],
                                   [0.021    , 0.343    ],
                                   [0.058    , 0.292    ],
                                   [0.153    , 0.295    ]]))
    
    #TODO upload selection file (change filename accordingly)
    (inds1, inds2, sf1, sf2, sample2d) = load_pickle("gms_selection")
    
    
    #%% run all selected ground motions with opensees

    ims_g1 = list()
    ims_g2 = list()
    hysts_g1 = list()
    hysts_g2 = list()
    maxds_g1 = list()
    maxds = list()
    for i, (ind1, ind2, s1, s2) in enumerate(tqdm(zip(inds1, inds2, sf1, sf2))):
        rp = RecordPair(rc.records[ind1], rc.records[ind2], s1, s2, 40., 10.)
        _ = opr.run(rp.get_time_gmr(unit="m/s2"))
        out = PSDM_gg21.extract_lists(rp, opr, sample2d[i,0], sample2d[i,1])
        ims_g1.append( out["im_g1"] )
        ims_g2.append( out["im_g2"] )
        hysts_g1.append( out["hyst_g1"] )
        hysts_g2.append( out["hyst_g2"] )
        maxds_g1.append( out["maxd_g1"] )
        maxds.append( out["maxd"] )
    save_pickle((ims_g1, ims_g2, hysts_g1, hysts_g2, maxds_g1, maxds), "response")
        
