# -*- coding: utf-8 -*-
"""
@author: Salvatore
"""


class Fun:
    
    def __init__(self, psdm, dsc):
        self.psdm = psdm
        self.dsc = dsc
        

    def fitness_pair_nga(self, x):
        '''
        x is a 1d numpy array:
        [index GM1 * n, index GM2 * n, scaling GM1 * n, scaling GM2 * n]
        '''
        self.psdm.a = x[0]
        self.psdm.b = x[1]
        self.psdm.c0 = x[2]
        self.psdm.d = x[3]
        self.psdm.m = x[4]
        ddfc = DamgDepFragCurvesNoC(self.psdm, self.dsc)
        print(ddfc.beta)
        return ddfc.beta


