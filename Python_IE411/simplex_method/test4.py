# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 21:39:46 2017

@author: Siqi Miao
"""
# test4.py
#
# Bland's rule pivot rule test.
#
# indices of iB, iN start with 1

import numpy as np
from numpy.linalg import norm
from simplex_step import simplex_step

# start with a tableau form
A1 = np.matrix([[1,    1,     2],
                [1,    1,     1],
                [1,    1,     1]], dtype=np.float64)

A = np.hstack((np.eye(3), A1))


b = np.matrix([[1],
               [2],
               [3]], dtype=np.float64)


iB = [1, 2, 3]
iN = [4, 5, 6]
xB = np.matrix(np.copy(b))
c = np.matrix([[0, 0, 0, -1, -2, 1]], dtype=np.float64)

# form an invertible matrix B and modify the problem
B = np.matrix([[4, 1, 0],
               [1, -2, -1],
               [1, 2, 4]], dtype=np.float64)
A = B*A
b = B*b

# modify c
N = A[:, [index_N-1 for index_N in iN]]
c1 = np.matrix([[1, 1, 1]], dtype=np.float64)
c2 = c1*B.I*N+c[:, [index_N-1 for index_N in iN]]


# take a step with Bland's rule
irule = 1
[istatus, iB, iN, xB, Binv] = simplex_step(A, b, c, iB, iN, xB, irule=1)

X = np.zeros((6, 1), dtype=np.float64)
X[[(b-1) for b in iB]] = xB

if (istatus != 0):
    print('INCORRECT ISTATUS!\n')

if (norm(X-np.matrix([[0], [1], [2], [1], [0], [0]]) > 1e-10)):
    print('INCORRECT STEP!\n')


if (norm(np.array(sorted(iN))-np.array([1, 5, 6])) > 1e-10):
    print('iN incorrect!\n')


if (norm(np.array(sorted(iB))-np.array([2, 3, 4])) > 1e-10):
    print('iB incorrect!\n')
