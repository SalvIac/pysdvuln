# -*- coding: utf-8 -*-
"""
@author: Salvatore Iacoletti
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import openseespy.opensees as ops
from pysdvuln.base_sdof import BaseSDOF, get_sinewave
bsd = BaseSDOF()


if __name__ == '__main__':

    period = 1.0
    mass = bsd.MASS # unit (kg)
    f_yield = 150*bsd.MASS # unit (N)
    r_post = 1e-9
    k_spring = 4 * np.pi ** 2 * mass / period ** 2


    #%% simple sinewave

    ops.wipe()
    mat_props = [f_yield, k_spring, r_post]
    bsd.material(mat_props)
    bsd.geometry()
    
    time, sinewave = get_sinewave()
    outputs = bsd.cyclic_loading_analysis(sinewave)

    fig, axs = plt.subplots(2,1)
    axs[0].plot(outputs["disp"], outputs["force"], marker="o")
    axs[0].set_xlabel("Displacement (m)")
    axs[0].set_ylabel("Base shear (N)")
    axs[1].plot(time, sinewave, marker="o")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Displacement (m)")
    plt.show()


    #%% incremental sinewave with animation

    ops.wipe()
    mat_props = [f_yield, k_spring, r_post]
    bsd.material(mat_props)
    bsd.geometry()

    ampls = np.linspace(0,10,len(time))
    time, sinewave = get_sinewave(amplitude=ampls)
    outputs = bsd.cyclic_loading_analysis(sinewave)
    
    fig, axs = plt.subplots(2,1)
    axs[0].plot(outputs["disp"], outputs["force"], marker="o", markersize=2)
    axs[0].set_xlabel("Displacement (m)")
    axs[0].set_ylabel("Base shear (N)")
    axs[1].plot(time, sinewave, marker="o", markersize=2)
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Displacement (m)")
    plt.show()
    
    
    # animation
    fig, axs = plt.subplots(2,1)
    axs[0].set_xlim(-10,10)
    axs[0].set_ylim(-f_yield*1.1,f_yield*1.1)
    axs[0].set_xlabel("Displacement (m)")
    axs[0].set_ylabel("Base shear (N)")
    axs[1].set_xlim(0,10)
    axs[1].set_ylim(-10,10)
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Displacement (m)")
    def animate(i):
        axs[0].plot(outputs["disp"][:i], outputs["force"][:i], marker="o", 
                    color="b", markersize=2)
        axs[1].plot(time[:i], sinewave[:i], marker="o", color="b", markersize=2)
    anim = FuncAnimation(fig, animate,# init_func=init,
                         frames=outputs["disp"].shape[0],
                         interval=outputs["disp"].shape[0]/10)
    anim.save('cyclic.gif', writer='imagemagick')


