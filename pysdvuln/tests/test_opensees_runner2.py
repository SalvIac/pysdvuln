# -*- coding: utf-8 -*-
"""
@author: Salvatore Iacoletti
"""

import os
import time
from pysdvuln.record import Record
from pysdvuln.opensees_runner import OpenseesRunner


if __name__ == "__main__":
    
    base = os.path.join(os.getcwd(), "sample_ground_motions")

    # nga west2
    path = os.path.join(base, "NGAWest2", "time_histories", "RSN6_IMPVALL.I_I-ELC180.AT2")
    rec = Record.load_ngawest2_record(path, allinfo=False)
    # rec.plot_inputs()
    # rec.plot_acc_vel_disp(unit="g")

    path = r"sample_capacity_curves"
    opr = OpenseesRunner.from_gem(path, "CR_LFINF-DUL_H2")
    # opr.DEGRADATION = False
    # opr.COLLAPSE_FACTOR = 1.
    t = time.time()
    res = opr.run(rec.get_time_gmr(unit="m/s2", resting_time=0))
    print(time.time()-t)
    opr.plot_acc_disp_th()
    opr.plot_force_disp_wth(ax2="disp")
    
    print(opr.get_max_disp(), opr.calculate_hyst_energy())
    
