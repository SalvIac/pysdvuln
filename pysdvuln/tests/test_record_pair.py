# -*- coding: utf-8 -*-
"""
@author: Salvatore Iacoletti
"""

import os
from pysdvuln.record import Record
from pysdvuln.record_pair import RecordPair


if __name__ == "__main__":
    
    base = os.path.join(os.getcwd(), "sample_ground_motions")

    # simbad
    path = os.path.join(base, "SIMBAD (Updated 2011)", "Records", "IN0001xa_record.mat")
    spectrum_path = os.path.join(base, "SIMBAD (Updated 2011)", "Spectra", "IN0001xa.mat")
    rec1 = Record.load_simbad_record(path, allinfo=True, spectrum_path=spectrum_path)
    rec1.plot_inputs()
    
    # nga west2
    path = os.path.join(base, "NGAWest2", "time_histories", "RSN6_IMPVALL.I_I-ELC180.AT2")
    rec2 = Record.load_ngawest2_record(path, allinfo=False)
    rec2.plot_inputs()


    rp = RecordPair(rec1, rec2, sf1=1.5, sf2=0.5, lag=40., rest=10.)
    rp.plot_acc()
