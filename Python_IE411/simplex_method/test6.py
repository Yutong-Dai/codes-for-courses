# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 21:02:17 2017

@author: Siqi Miao
"""

# test6.py
#
# First initialization test: infeasible problem.
#
# indices of iB, iN start with 1


import numpy as np
from simplex_init import simplex_init

A = np.matrix([[1, 1, 1, 2,  1, 3],
               [1, 1, 0, 2,  2, 2],
               [1, 0, 0, 12, 1, 1]], dtype=np.float64)

b = np.matrix([[-1],
               [3],
               [-1]], dtype=np.float64)

c = np.matrix([[-1, -1, -1, -1, -1, -1]], dtype=np.float64)

[istatus, iB, iN, xB] = simplex_init(A, b, c)

if (istatus != 16):
    print('istauts WRONG!!!!\n')

A = np.matrix(np.copy(-A))
c = np.matrix(np.copy(c))

[istatus, iB, iN, xB] = simplex_init(A, b, c)

if (istatus != 16):
    print('istauts WRONG!!!!\n')
