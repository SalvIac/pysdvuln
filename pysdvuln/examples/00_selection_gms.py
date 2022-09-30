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
import matplotlib.pyplot as plt
from pysdvuln.utils_pickle import save_pickle, load_pickle
from pysdvuln.records_container import RecordsContainer
from pysdvuln.selector import Selector

'''
select ground motions from database
database of candidate ground motions not provided
look for #TODO to customize
'''

if __name__ == "__main__":
    
    #TODO define inputs
    n = 1200
    max_im = 3. # in g
    imt = "AvgSa" # can be PGA, SA or AvgSa
    period = 0.4 # in s (relevant if imt is SA or AvgSa)
    
    
    #%% create database
    
    #TODO define database of candidate ground motions
    # note that the database used for Iacoletti et al. (2023) is not provided
    base = r"C:\Users\Salvatore\Dropbox\SalvIac\ground_motion_databases"
    rc = RecordsContainer.load_ngawest2_data(base)
    for rec in tqdm(rc.records):
        spec = rec.generate_response_spectrum(np.array([0.001, 6]))
        rec.precomp_spectrum = np.vstack([spec[0,:], rec.precomp_spectrum, spec[-1,:]])
    rsns = rc.get_array("rsn")
    print(len(rc.records))
    
    # save database for later
    save_pickle(rc, "database_gms")
    # rc = load_pickle("database_gms") # in case you already run the above
    
    
    #%% define selector
    
    if imt == "PGA":
        imt2 = imt
        ims = rc.get_pgas()
    elif imt == "SA":
        imt2 = "SA({:.2f}s)".format(period)
        ims = rc.get_sas(period)
    elif imt == "AvgSa":
        imt2 = "AvgSa({:.2f}s)".format(period)
        ims = rc.get_avgsas_T(period)
    else:
        raise Exception("check imt")
    print("maximum IM in catalog", np.max(ims)/9.81, "g")
    rsns = rc.get_array("rsn")

    sel = Selector(ims, rsns, min_im=0., max_im=max_im*9.81, bins=20, imt=imt2)
    
    
    #%% simulating annealing selection
    
    inds1, inds2, sf1, sf2, sample2d = sel.simul_ann(n, minscale=0.5, maxscale=2.,
                                                      maxfun=1e7, maxiter=1000)
    save_pickle((inds1, inds2, sf1, sf2, sample2d), "gms_selection")


    #%% some plots
    
    # scatter plot scaled ims (in m/s2) from selected ground motions
    ims1 = list()
    ims2 = list()
    for i, (ind1, ind2, s1, s2) in enumerate(tqdm(zip(inds1, inds2, sf1, sf2))):
        rec1 = rc.records[ind1].scale(s1)
        rec2 = rc.records[ind2].scale(s2)
        if imt == "PGA":
            ims = rc.get_pgas()
            ims1.append( rec1.get_sa(0.) )
            ims2.append( rec2.get_sa(0.) )
        elif imt == "SA":
            ims1.append( rec1.get_sa_T(period) )
            ims2.append( rec2.get_sa_T(period) )
        elif imt == "AvgSa":
            ims1.append( rec1.get_avgsa_T(period) )
            ims2.append( rec2.get_avgsa_T(period) )
        else:
            raise Exception("check imt")
    plt.figure()
    plt.scatter(ims1, ims2)
    plt.show()
    
    # or simply
    sel.scatter(sample2d[:,0], sample2d[:,1])
    