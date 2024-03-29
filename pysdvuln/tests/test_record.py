# -*- coding: utf-8 -*-
"""
@author: Salvatore Iacoletti
"""

import os
import numpy as np
from pysdvuln.record import Record
from pysdvuln.record_two_comp import RecordTwoComp
from pysdvuln.base_sdof import get_sinewave


if __name__ == "__main__":

    base = os.path.join(os.getcwd(), "sample_ground_motions")

    #%% simple sinewave
    
    dt = 1e-2
    _, record = get_sinewave(amplitude=1., dt=dt)
    rec = Record(record, dt)
    rec.plot_inputs()
    rec.plot_response_spectrum()
    rec.plot_acc_vel_disp()


    #%% simbad
    
    path = os.path.join(base, "SIMBAD (Updated 2011)", "Records", "IN0001xa_record.mat")
    spectrum_path = os.path.join(base, "SIMBAD (Updated 2011)", "Spectra", "IN0001xa.mat")
    rec = Record.load_simbad_record(path, allinfo=True, spectrum_path=spectrum_path)
    rec.plot_inputs()
    rec.plot_response_spectrum()
    rec.plot_acc_vel_disp()
    rec.plot_comparison_spectra()
    rec.plot_husid(start=0.001, end=0.999)
    
    
    #%% goda

    path = os.path.join(base, "MS-AS sequences by Katsu Goda", "MS_H1_1.dat")
    rec = Record.load_goda_record(path, allinfo=True)
    rec.plot_inputs()
    rec.plot_response_spectrum()
    rec.plot_acc_vel_disp()
    rec.plot_comparison_spectra()
    rec.plot_husid(start=0.05, end=0.95)
    
    # scaled goda
    rec2 = rec.scale(0.5)
    rec2.plot_inputs()
    rec2.plot_response_spectrum()
    rec2.plot_acc_vel_disp()
    rec2.plot_comparison_spectra()
    rec.plot_husid(start=0.05, end=0.95)
    
    
    #%% nga west2

    two_up = os.path.join(base, "NGAWest2")
    metadata = Record.load_ngawest2_flatfile(two_up)

    path = os.path.join(base, "NGAWest2", "time_histories", "RSN6_IMPVALL.I_I-ELC180.AT2")
    rec = Record.load_ngawest2_record(path, allinfo=True, metadata=metadata)
    rec.plot_inputs()
    rec.plot_response_spectrum()
    rec.get_sa(0.)
    rec.get_sa(0.5)
    rec.get_avgsa_T(0.5)
    rec.plot_husid(start=0.05, end=0.95)


    # check velocity and displacement are OK
    axs = rec.plot_acc_vel_disp()

    pathv = path.replace("AT2", "VT2")
    with open(pathv) as f:
        lines = f.readlines()
    gvr = list()
    for line in lines[4:]:
        for el in line.replace("\n","").split(" "):
            if el == "" or el.isspace():
                continue
            gvr.append(float(el))
    gvr = np.array(gvr)/1e2
    axs[1].plot(rec.time, gvr, linestyle="--")

    pathd = path.replace("AT2", "DT2")
    with open(pathd) as f:
        lines = f.readlines()
    gdr = list()
    for line in lines[4:]:
        for el in line.replace("\n","").split(" "):
            if el == "" or el.isspace():
                continue
            gdr.append(float(el))
    gdr = np.array(gdr)/1e2
    axs[2].plot(rec.time, gdr, linestyle="--")
    
    
    #%% 2 horizontal components of the nga west2
    
    # from searchResults_geomean
    periods = [0.01,0.02,0.022,0.025,0.029,0.03,0.032,0.035,0.036,0.04,0.042,0.044,0.045,0.046,0.048,0.05,0.055,0.06,0.065,0.067,0.07,0.075,0.08,0.085,0.09,0.095,0.1,0.11,0.12,0.13,0.133,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.22,0.24,0.25,0.26,0.28,0.29,0.3,0.32,0.34,0.35,0.36,0.38,0.4,0.42,0.44,0.45,0.46,0.48,0.5,0.55,0.6,0.65,0.667,0.7,0.75,0.8,0.85,0.9,0.95,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2,2.2,2.4,2.5,2.6,2.8,3,3.2,3.4,3.5,3.6,3.8,4,4.2,4.4,4.6,4.8,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,11,12,13,14,15,20]
    check = np.array([0.10168804,0.10270724,0.10386547,0.10379523,0.10417092,0.10382875,0.10611354,0.10966214,0.11036027,0.1070665,0.11027225,0.11769825,0.12118102,0.12212646,0.11782187,0.11232865,0.12532451,0.13256471,0.1353921,0.1470277,0.16850198,0.1652963,0.1505367,0.14981357,0.14694565,0.1536317,0.14507829,0.15034408,0.18482484,0.17846054,0.17527399,0.16510585,0.1804494,0.1994142,0.22999843,0.24752623,0.28870002,0.27353183,0.2137812,0.19581369,0.21736164,0.23815334,0.25851117,0.25605851,0.24704849,0.23111338,0.2053543,0.19419293,0.18459587,0.18573231,0.18807721,0.18463887,0.17841533,0.17581989,0.17229759,0.16952781,0.17223278,0.20151243,0.2439786,0.27137809,0.27487259,0.27396189,0.25804454,0.23052255,0.1964954,0.16981481,0.16199493,0.15008761,0.10822356,0.09235156,0.08902386,0.08942364,0.09054697,0.0871967,0.07603562,0.0676486,0.05943408,0.05138638,0.04039687,0.03442121,0.03207312,0.02962579,0.02518645,0.02166141,0.01899648,0.0172411,0.01608938,0.01525724,0.01341181,0.01105808,0.00958618,0.00849047,0.00728741,0.0061176,0.00538397,0.00408918,0.00332358,0.00280479,0.00238853,0.00205009,0.00177338,0.00154582,0.00135729,0.00119982,0.00106729,0.00085903,0.00070529,0.00058897,0.00049905,0.0004282,0.0002288])
 
    # not all record components have the same number of points
    path1 = os.path.join(base, "NGAWest2", "time_histories", "RSN130_FRIULI.B_B-BUI000.AT2")
    path2 = os.path.join(base, "NGAWest2", "time_histories", "RSN130_FRIULI.B_B-BUI270.AT2")

    rec1 = Record.load_ngawest2_record(path1, allinfo=True, metadata=metadata)
    rec2 = Record.load_ngawest2_record(path2, allinfo=True, metadata=metadata)
    trec = RecordTwoComp(rec1, rec2)
    
    ax = trec.plot_response_spectrum(periods)    
    ax.plot(periods, check*9.81, color="k", linestyle="--")
    ax.set_xscale("log")
    ax.set_ylim([0,0.45*9.81])
    
    trec.plot_inputs()
    trec.plot_response_spectrum()
    trec.get_sa(0.)
    trec.get_sa(0.5)
    trec.get_avgsa_T(0.5)
    
    
    