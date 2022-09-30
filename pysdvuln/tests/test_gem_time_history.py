# -*- coding: utf-8 -*-
"""
@author: Salvatore Iacoletti
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import eqsig
import openseespy.opensees as ops
from pysdvuln.base_sdof import BaseSDOF
bsd = BaseSDOF()


if __name__ == '__main__':

    filepath = "sample_capacity_curves/CR_LFINF-DUL_H2.csv" 
    capacity_curve = pd.read_csv(filepath).to_numpy()
    # capacity_curve[:,1] = capacity_curve[:,1]/3
    
    degradation = False
    damping = 0.05

    record_filename = 'sample_ground_motions/test_motion_dt0p01.txt'
    motion_step = 0.01
    rec = np.loadtxt(record_filename) #*3
    ground_motion_ms2 = np.vstack([np.arange(0., len(rec) * motion_step, motion_step),
                                   rec]).transpose()
    ground_motion_g = np.vstack([np.arange(0., len(rec) * motion_step, motion_step),
                                 -rec/9.81]).transpose()


    #%% GEM functions
    
    # import sys
    # sys.path.append(r'C:\Users\Salvatore\Dropbox\SalvIac\VMTK-Vulnerability-Modellers-ToolKit\analysis')
    # from nlth_on_sdof import run_nlth_analysis_on_sdof_ops_py

    # times, disps, accels, forces = run_nlth_analysis_on_sdof_ops_py(capacity_curve, 
    #                                                         ground_motion_g,
    #                                                         damping,
    #                                                         degradation)


    #%% my functions
    
    period = bsd.get_secant_period(capacity_curve)
    
    ops.wipe()
    bsd.material_martins_silva(capacity_curve, degradation=degradation)
    bsd.geometry()
    tt = time.time()
    outputs = bsd.time_history_analysis(ground_motion_ms2, damping)
    print(time.time() - tt)


    #%% elastic response with eqsig
    
    periods = np.array([period])
    acc_signal = eqsig.AccSignal(ground_motion_ms2[:,1], motion_step)
    resp_u, resp_v, resp_a = eqsig.sdof.response_series(motion=ground_motion_ms2[:,1], 
                                                        dt=motion_step,
                                                        periods=periods,
                                                        xi=damping)
    

    #%% plot comparisons
    
    fig, axs = plt.subplots(nrows=2, sharex=True)
    axs[0].plot(acc_signal.time, resp_u[0], label="Eqsig", lw=1, color="k")
    axs[0].plot(outputs["time"], outputs["disp"], label="Opensees", ls="--", lw=1, color="b")
    # axs[0].plot(times, disps, label="GEM", ls=":", lw=1, color="r")
    axs[0].set_ylabel("Displacement (m)")
    axs[1].plot(acc_signal.time, resp_a[0]+rec, label="Eqsig", lw=1, color="k")  # Elastic solution
    # acc_opensees_elastic = np.interp(acc_signal.time, ) - rec
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
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Displacement (m)")
    plt.show()
    
