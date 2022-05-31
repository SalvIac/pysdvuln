# -*- coding: utf-8 -*-
"""
@author: Salvatore Iacoletti
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import openseespy.opensees as ops
from pysdvuln.base_sdof import BaseSDOF, get_sinewave
bsd = BaseSDOF()


if __name__ == '__main__':
    
    filepath = "sample_capacity_curves/CR_LFINF-DUL_H2.csv" 
    capacity_curve = pd.read_csv(filepath).to_numpy()
    degradation = True
    
    
    #%% simple sinewave just below failure point
    
    ops.wipe()
    bsd.material_martins_silva(capacity_curve, degradation=degradation)
    bsd.geometry()
    
    time, sinewave = get_sinewave(amplitude=capacity_curve[-1,0]*0.9, dt=1e-2)
    outputs = bsd.cyclic_loading_analysis(sinewave)

    fig, axs = plt.subplots(2,1)
    axs[0].plot(capacity_curve[:,0], capacity_curve[:,1]*9.81*bsd.MASS, marker="o", markersize=2)
    axs[0].plot(outputs["disp"], outputs["force"], marker="o", markersize=2)
    axs[0].set_xlabel("Displacement (m)")
    axs[0].set_ylabel("Base shear (N)")
    axs[1].plot(time, sinewave, marker="o")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Displacement (m)")
    plt.show()


    #%% simple sinewave just above failure point
    
    ops.wipe()
    bsd.material_martins_silva(capacity_curve, degradation=degradation)
    bsd.geometry()
    
    time, sinewave = get_sinewave(amplitude=capacity_curve[-1,0]*1.1, dt=1e-2)
    outputs = bsd.cyclic_loading_analysis(sinewave)

    fig, axs = plt.subplots(2,1)
    axs[0].plot(capacity_curve[:,0], capacity_curve[:,1]*9.81*bsd.MASS, marker="o", markersize=2)
    axs[0].plot(outputs["disp"], outputs["force"], marker="o", markersize=2)
    axs[0].set_xlabel("Displacement (m)")
    axs[0].set_ylabel("Base shear (N)")
    axs[1].plot(time, sinewave, marker="o", markersize=2)
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Displacement (m)")
    plt.show()


    #%% incremental sinewave with animation
    
    ops.wipe()
    bsd.material_martins_silva(capacity_curve, degradation=degradation)
    bsd.geometry()

    ampls = np.linspace(0, capacity_curve[-1,0]*1.2, len(time))
    time, sinewave = get_sinewave(amplitude=ampls, dt=1e-2)
    outputs = bsd.cyclic_loading_analysis(sinewave)
    
    fig, axs = plt.subplots(2,1)
    axs[0].plot(capacity_curve[:,0], capacity_curve[:,1]*9.81*bsd.MASS, marker="o", markersize=2)
    axs[0].plot(outputs["disp"], outputs["force"], marker="o", markersize=2)
    axs[0].set_xlabel("Displacement (m)")
    axs[0].set_ylabel("Base shear (N)")
    axs[1].plot(time, sinewave, marker="o", markersize=2)
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Displacement (m)")
    plt.show()
    
    
    # animation
    fig, axs = plt.subplots(2,1)
    axs[0].set_xlim(-capacity_curve[-1,0]*1.1, capacity_curve[-1,0]*1.1)
    axs[0].set_ylim(-capacity_curve[-1,1]*1.1*9.81*bsd.MASS, capacity_curve[-1,1]*1.1*9.81*bsd.MASS)
    axs[0].set_xlabel("Displacement (m)")
    axs[0].set_ylabel("Base shear (N)")
    axs[1].set_xlim(0,10)
    axs[1].set_ylim(-capacity_curve[-1,0]*1.2,capacity_curve[-1,0]*1.2)
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Displacement (m)")
    axs[0].plot(capacity_curve[:,0], capacity_curve[:,1]*9.81*bsd.MASS, marker="o",
                color="r", markersize=2)
    def animate(i):
        axs[0].plot(outputs["disp"][:i], outputs["force"][:i], marker="o",
                    color="b", markersize=2)
        axs[1].plot(time[:i], sinewave[:i], marker="o", color="b", 
                    markersize=2)
    anim = FuncAnimation(fig, animate,# init_func=init,
                         frames=outputs["disp"].shape[0],
                         interval=outputs["disp"].shape[0]/10)
    plt.show()
    
    ss
    anim.save('cyclic2.gif', writer='imagemagick')
        
        
    
