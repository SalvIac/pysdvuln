# -*- coding: utf-8 -*-
"""
@author: Salvatore Iacoletti
"""

from pysdvuln.damg_state_classifier import DamgStateClassifier


if __name__ == "__main__":

    thresholds = [0.03, 0.19, 1.32, 1.76]
    dsc = DamgStateClassifier(thresholds=thresholds)
    
    print(dsc.classify([0.02, 0.05, 1., 1.5, 2.]))
    print(dsc.get_damage_states())
    print(dsc.get_damage_states_str())
    