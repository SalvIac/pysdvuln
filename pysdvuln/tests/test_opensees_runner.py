# -*- coding: utf-8 -*-
"""
@author: Salvatore Iacoletti
"""

import os
import time
import numpy as np
from pysdvuln.record import Record
from pysdvuln.opensees_runner import OpenseesRunner


if __name__ == "__main__":
    
    record_filename = 'sample_ground_motions/test_motion_dt0p01.txt'
    motion_step = 0.01
    rec = np.loadtxt(record_filename)
    rec2 = Record(rec, motion_step, unit="m/s2")
    # rec2.plot_inputs()
    # rec2.plot_acc_vel_disp(unit="g")

    path = r"sample_capacity_curves"
    opr = OpenseesRunner.from_oq(path, "CR_LFINF-DUL_H2")
    opr.DEGRADATION = False
    t = time.time()
    res = opr.run(rec2.get_time_gmr(unit="m/s2", resting_time=0))
    print(time.time()-t)
    opr.plot_acc_disp_th()
    opr.plot_force_disp_wth(ax2="disp")
   