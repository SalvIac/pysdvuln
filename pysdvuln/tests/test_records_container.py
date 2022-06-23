# -*- coding: utf-8 -*-
"""
@author: Salvatore Iacoletti
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from pysdvuln.records_container import RecordsContainer


if __name__ == "__main__":
   
    base = os.path.join(os.getcwd(), "sample_ground_motions")

    # rc = RecordsContainer.load_ngawest2_data(base)
    rc = RecordsContainer.load_simbad_data(base)
    # rc = RecordsContainer.load_goda_data(base)

    
    # rc = RecordsContainer.load_ngawest2_data_comps(base)
    # records = list()
    # for recs in rc.records:
    #     records.append( recs.choose(recs.rec1.get_avgsa(0.5),
    #                                 recs.rec2.get_avgsa(0.5)) )
    # rc = RecordsContainer(records, database=rc.database, sourcepath=rc.sourcepath)


    #%% filters
    
    # rsns = rc.get_array("rsn")
    # df = pd.read_csv(os.path.join(base, "NGAWest2", "pulse_like_peer.csv"))
    # pulse_rsn = df['Record Seq. #'].to_list()
    # del_i = [i for i, rsn in enumerate(rsns) if rsn in pulse_rsn]
    # print(len(rc.records))
    # rc.delete_records(del_i)
    
    print(len(rc.records))
    rc.filter_sign_duration(1)
    print(len(rc.records))
    
    
    # from tqdm import tqdm
    # for rec in tqdm(rc.records[1800:2000]):
    #     rec.plot_inputs()
    
    # todelete = rc.check_duplicates()
    # rc.delete_records(inds)
    
    
    #%%
    
    durations = rc.get_durations()
    fig = plt.figure(figsize=(6,6))
    plt.hist(durations, bins=50)
    plt.xlabel("duration (s)")
    plt.ylabel("histogram")
    plt.show()

    durations = rc.get_sign_durations()
    fig = plt.figure(figsize=(6,6))
    plt.hist(durations, bins=np.arange(0,360,2))
    plt.vlines(2, 0, 900, colors="r")
    plt.ylim([0,900])
    plt.xlabel("significant duration (s)")
    plt.ylabel("histogram")
    plt.show()
    
    mags = rc.get_array("magnitude")
    dists = rc.get_array("epicentral_distance")
    fig = plt.figure(figsize=(6,6))
    plt.scatter(dists, mags, s=5, edgecolor='k', linewidth=0.1)
    plt.xlabel("epicentral distance")
    plt.ylabel("magnitude")
    plt.xscale("log")
    plt.show()

    mags = rc.get_array("magnitude")
    dists = rc.get_array("epicentral_distance")
    fig = plt.figure(figsize=(6,6))
    plt.scatter(dists, mags, s=5, edgecolor='k', linewidth=0.1)
    plt.xlabel("epi distance")
    plt.ylabel("magnitude")
    plt.xscale("log")
    plt.show()

    vs30s = rc.get_array("vs30")
    fig = plt.figure(figsize=(6,6))
    plt.hist(vs30s, bins=20)
    plt.xlabel("vs30")
    plt.ylabel("histogram")
    plt.show()

    classes = rc.get_array("site_class")
    classes_int = np.zeros(classes.shape)
    classes_int[np.where(classes=="A")[0]] = 0
    classes_int[np.where(classes=="B")[0]] = 1
    classes_int[np.where(classes=="C")[0]] = 2
    classes_int[np.where(classes=="D")[0]] = 3
    classes_int[np.where(classes=="E")[0]] = 4
    fig = plt.figure(figsize=(6,6))
    plt.hist(classes_int, bins=[-0.5,0.5,1.5,2.5,3.5,4.5])
    plt.xlabel("classes")
    plt.ylabel("histogram")
    ax = plt.gca()
    ax.set_xticks(range(5))
    ax.set_xticklabels(["A","B","C","D","E"])
    plt.show()
    
    
    #%%
    
    r0 = rc.get_sas(0.)
    r001 = rc.get_sas(0.01)
    r02 = rc.get_sas(0.2)
    r05 = rc.get_sas(0.5)
    ravg = rc.get_avgsas_T(0.5)
    ims = ravg
