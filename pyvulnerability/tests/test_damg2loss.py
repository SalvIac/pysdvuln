# -*- coding: utf-8 -*-
"""
@author: Salvatore
"""

from pysdvuln.damg2loss import Damg2Loss


d2l = Damg2Loss.default()
d2l.check()
print(d2l.get_mean_loss())
