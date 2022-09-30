# -*- coding: utf-8 -*-
"""
@author: Salvatore Iacoletti
"""

import pandas as pd
import matplotlib.pyplot as plt
import openseespy.opensees as ops
from pysdvuln.base_sdof import BaseSDOF
bsd = BaseSDOF()


if __name__ == '__main__':

    ops.wipe()
    filepath = "sample_capacity_curves/CR_LFINF-DUL_H2.csv" 
    capacity_curve = pd.read_csv(filepath).to_numpy()
    bsd.material_martins_silva(capacity_curve)
    bsd.geometry()
    
    outputs = bsd.pushover_analysis(capacity_curve[-1,0]*1.1, capacity_curve[-1,0]/100)

    fig, ax = plt.subplots(1,1)
    ax.plot(capacity_curve[:,0], capacity_curve[:,1]*9.81*bsd.MASS, marker="o")
    ax.plot(outputs["disp"], outputs["force"], marker="o")
    ax.set_xlabel("Displacement (m)")
    ax.set_ylabel("Base shear (N)") # technically base shear per unit mass
    plt.show()

