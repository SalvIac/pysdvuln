# -*- coding: utf-8 -*-
"""
@author: Salvatore Iacoletti
"""

import numpy as np
import matplotlib.pyplot as plt
import openseespy.opensees as ops
from pysdvuln.base_sdof import BaseSDOF
bsd = BaseSDOF()


if __name__ == '__main__':

    period = 1.0
    mass = bsd.MASS
    f_yield = 150*bsd.MASS # (unit: N)
    r_post = 1e-9
    k_spring = 4 * np.pi ** 2 * mass / period ** 2

    ops.wipe()
    mat_props = [f_yield, k_spring, r_post]
    bsd.material(mat_props)
    bsd.geometry()
    
    outputs = bsd.pushover_analysis(10, 0.1)

    fig, ax = plt.subplots(1,1)
    ax.plot(outputs["disp"], outputs["force"], marker="o")
    ax.set_xlabel("Displacement (m)")
    ax.set_ylabel("Base shear (N)") # technically base shear per unit mass if MASS=1.
    plt.show()
