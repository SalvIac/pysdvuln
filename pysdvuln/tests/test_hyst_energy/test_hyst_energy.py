# -*- coding: utf-8 -*-
"""
@author: Salvatore Iacoletti
"""

import os
import time
import numpy as np
import pandas as pd
from pysdvuln.record import Record
from pysdvuln.opensees_runner import OpenseesRunner
from pysdvuln.base_sdof import BaseSDOF
bsd = BaseSDOF()


if __name__ == "__main__":
    '''
    the same results are obtained by using mass_eff=1., dividing the 
    capacity curve by 907.6432850690459 (weight in kN) to obtain the 
    hysteretic energy in kNm per unit mass. Then multiply kNm per unit 
    mass times the # mass (907.6432850690459/9.81) to get the hysteretic
    energy
    '''
    mass_eff = 907.6432850690459/9.81 # ton
    
    gmr = np.loadtxt("INPUT.txt")
    gmr1 = gmr[gmr[:,0]<=40.001,:]
    rec1 = Record(gmr1[:,1], t_step=0.005, unit="g")
    rec2 = Record(gmr[:,1], t_step=0.005, unit="g")
    # rec.plot_inputs()
    # rec.plot_acc_vel_disp(unit="g")
    
    capacity_curve = pd.read_csv("test.csv").to_numpy()
    # Livio has capacity curves in kN, I need them in g (divide by mass_eff and 9.81)
    capacity_curve[:,1] = capacity_curve[:,1]/9.81/mass_eff # /907.6432850690459
    
    opr = OpenseesRunner(capacity_curve, mass_eff)
    opr.DEGRADATION = False
    
    # mainshock
    res = opr.run(rec1.get_time_gmr(unit="m/s2", resting_time=0), mat="Hysteretic")
    # opr.plot_acc_disp_th()
    # opr.plot_force_disp_wth()
    opr.plot_force_disp()
    print(opr.get_fund_period(), opr.get_max_disp(), opr.calculate_hyst_energy())

    # mainshock + aftershock
    res = opr.run(rec2.get_time_gmr(unit="m/s2", resting_time=10), mat="Hysteretic")
    # opr.plot_acc_disp_th()
    # opr.plot_force_disp_wth()
    opr.plot_force_disp()
    print(opr.get_fund_period(), opr.get_max_disp(), opr.calculate_hyst_energy())
    