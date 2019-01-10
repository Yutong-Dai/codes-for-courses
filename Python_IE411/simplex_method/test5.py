# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 21:39:46 2017

@author: Siqi Miao
"""
# test5.py
#
# Unboundedness test.
#
# indices of iB, iN start with 1

import numpy as np
from simplex_step import simplex_step

# start with a tableau form
A1 = np.matrix([[-1,    1,     2],
                [-1,    1,     1],
                [0,    1,     1]], dtype=np.float64)

A = np.hstack((np.eye(3), A1))


b = np.matrix([[1],
               [2],
               [3]], dtype=np.float64)


iB = [1, 2, 3]
iN = [4, 5, 6]
xB = np.matrix(np.copy(b))
c = np.matrix([[0, 0, 0, -1, 2, 1]], dtype=np.float64)

# form an invertible matrix B and modify the problem
B = np.matrix([[4, 1, 0],
               [1, -2, -1],
               [1, 2, 4]], dtype=np.float64)
A = B*A
b = B*b

# modify c
N = A[:, [index_N-1 for index_N in iN]]
c1 = np.matrix([[1, 1, 0]], dtype=np.float64)
c2 = c[:, (4-1):6]+c1*B.I*N
c = np.hstack((c1, c2))

# take a step.
irule = 0
[istatus, iB, iN, xB, Binv] = simplex_step(A, b, c, iB, iN, xB, irule)


X = np.zeros((6, 1), dtype=np.float64)
X[[(b-1) for b in iB]] = xB

if (istatus != 16):
    print('INCORRECT ISTATUS!\n')
