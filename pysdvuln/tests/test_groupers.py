# -*- coding: utf-8 -*-
"""
@author: Salvatore
"""

from pysdvuln.groupers import BaseGrouper


if __name__ == '__main__':
    bg = BaseGrouper(centerbin=[1., 1.5, 2., 2.5, 3.],
                     bins=[ 0., 0.5, 0.75, 1., 1.25 ])
    print([bg(val) for val in [0.2, 0.7, 0.9, 1.2, 10]])
    

