# -*- coding: utf-8 -*-
"""
@author: Salvatore Iacoletti
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import eqsig
import openseespy.opensees as ops
from pysdvuln.record import Record
from pysdvuln.base_sdof import BaseSDOF
bsd = BaseSDOF()


if __name__ == "__main__":
    
    filepath = "sample_capacity_curves/CR_LFINF-DUL_H2.csv" 
    capacity_curve = pd.read_csv(filepath).to_numpy()
    
    degradation = True
    damping = 0.05

    # nga west2
    base = os.path.join(os.getcwd(), "sample_ground_motions")
    path = os.path.join(base, "NGAWest2", "time_histories", "RSN6_IMPVALL.I_I-ELC180.AT2")
    rec = Record.load_ngawest2_record(path, allinfo=False)
    # rec.plot_inputs()
    # rec.plot_acc_vel_disp(unit="g")
    ground_motion_g = rec.get_time_gmr(unit="g", resting_time=0)
    ground_motion_g[:,1] = -ground_motion_g[:,1]
    ground_motion_ms2 = rec.get_time_gmr(unit="m/s2", resting_time=0)
    

    #%% GEM functions
    
    # import sys
    # sys.path.append(r'C:\Users\Salvatore\Dropbox\SalvIac\VMTK-Vulnerability-Modellers-ToolKit\analysis')
    # from nlth_on_sdof import run_nlth_analysis_on_sdof_ops_py

    # times, disps, accels, forces = run_nlth_analysis_on_sdof_ops_py(capacity_curve, 
    #                                                         ground_motion_g,
    #                                                         damping,
    #                                                         degradation)


    #%% my functions
    
    period = bsd.get_fund_period(capacity_curve)

    ops.wipe()
    bsd.material_martins_silva(capacity_curve, degradation=degradation)
    bsd.geometry()
    tt = time.time()
    outputs = bsd.time_history_analysis(ground_motion_ms2, damping)
    print(time.time() - tt)


    #%% plot comparisons
    
    fig, axs = plt.subplots(nrows=2, sharex=True)
    axs[0].plot(outputs["time"], outputs["disp"], label="Opensees", ls="--", lw=1, color="b")
    # axs[0].plot(times, disps, label="GEM", ls=":", lw=1, color="r")
    axs[0].set_ylabel("Displacement {m}")
    axs[1].plot(outputs["time"], outputs["accel"], label="Opensees", ls="--", lw=1, color="b")
    # axs[1].plot(times, accels, label="GEM", ls=":", lw=1, color="r")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Acceleration (m/s2)")
    axs[1].legend()
    plt.show()


    fig, axs = plt.subplots(2,1)
    axs[0].plot(outputs["disp"], outputs["force"], lw=0.5, color="b")
    # axs[0].plot(disps, np.array(forces)*bsd.MASS, ls=":", lw=0.5, color="r")
    axs[0].plot(capacity_curve[:,0], capacity_curve[:,1]*9.81*bsd.MASS, lw=0.5, color="k")
    axs[0].set_xlabel("Displacement (m)")
    axs[0].set_ylabel("Base shear (N)")
    axs[1].plot(outputs["time"], outputs["disp"], ls="--", lw=0.5, color="b")
    # axs[1].plot(times, disps, ls=":", lw=0.5, color="r")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Displacement (m)")
    plt.show()
    