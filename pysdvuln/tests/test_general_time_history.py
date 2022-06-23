# -*- coding: utf-8 -*-
"""
@author: Salvatore Iacoletti
"""

import numpy as np
import matplotlib.pyplot as plt
import eqsig
import openseespy.opensees as ops
from pysdvuln.base_sdof import BaseSDOF
bsd = BaseSDOF()


if __name__ == '__main__':

    record_filename = 'sample_ground_motions/test_motion_dt0p01.txt'
    motion_step = 0.01
    rec = np.loadtxt(record_filename)
    ground_motion = np.vstack([np.arange(0., len(rec) * motion_step, motion_step),
                               rec]).transpose()
    acc_signal = eqsig.AccSignal(rec, motion_step)
    
    period = 1.0
    damping = 0.05
    mass = bsd.MASS # unit (kg)
    f_yield = 15.*bsd.MASS # unit (N)
    r_post = 0.
    
    
    periods = np.array([period])
    resp_u, resp_v, resp_a = eqsig.sdof.response_series(motion=rec, 
                                                        dt=motion_step,
                                                        periods=periods,
                                                        xi=damping)
    
    k_spring = 4 * np.pi ** 2 * mass / period ** 2
    omega = np.sqrt(k_spring/mass)


    ops.wipe()

    mat_props = [f_yield, k_spring, r_post]
    bsd.material(mat_props)
    bsd.geometry()

    outputs = bsd.time_history_analysis(ground_motion, damping)
    

    fig, axs = plt.subplots(nrows=2)
    axs[0].plot(acc_signal.time, resp_u[0], label="Eqsig")
    axs[0].plot(outputs["time"], outputs["disp"], label="Opensees fy=%.3gN" % f_yield, ls="--")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Displacement (m)")
    axs[1].plot(acc_signal.time, resp_a[0], label="Eqsig")  # Elastic solution
    time = acc_signal.time
    acc_opensees_elastic = np.interp(time, outputs["time"], outputs["accel"]) - rec
    axs[1].plot(time, acc_opensees_elastic, label="Opensees fy=%.2gN" % (f_yield), ls="--")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Acceleration (m/s2)")
    axs[1].legend()
    plt.show()


    fig, axs = plt.subplots(2,1)
    axs[0].plot(outputs["disp"], outputs["force"])
    axs[0].set_xlabel("Displacement (m)")
    axs[0].set_ylabel("Base shear (N)")
    axs[1].plot(outputs["time"], outputs["disp"])
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Displacement (m)")
    plt.show()
    