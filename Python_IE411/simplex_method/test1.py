# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 21:39:46 2017

@author: Siqi Miao
"""

# test1.py
#
# Test simplex_method.py by making sure that it takes a single
# step correctly.  This script uses a simple Tableau form.
#
# indices of iB, iN start with 1

import numpy as np
from numpy.linalg import norm
from simplex_step import simplex_step

# start with a Tableau form
A1 = np.matrix([[1, 1,  1],
                [1, 1, -1],
                [1, 1,  0]], dtype=np.float64)

A = np.hstack((np.eye(3), A1))

b = np.matrix([[1],
               [2],
               [3]], dtype=np.float64)


iB = [1, 2, 3]
iN = [4, 5, 6]
xB = np.matrix(np.copy(b))
c = np.matrix([[0, 0, 0, -1, 2, 1]], dtype=np.float64)


# test a step in this extremely simple state
irule = 0
[istatus, iB, iN, xB, Binv] = simplex_step(A, b, c, iB, iN, xB, irule)


X = np.zeros((6, 1), dtype=np.float64)
X[[(b-1) for b in iB]] = xB

if (istatus != 0):
    print('INCORRECT ISTATUS!\n')

if (norm(X-np.matrix([[0], [1], [2], [1], [0], [0]])) > 1e-10):
    print('INCORRECT STEP!\n')


if (norm(np.array(sorted(iN))-np.array([1, 5, 6])) > 1e-10):
    print('iN incorrect!\n')


if (norm(np.array(sorted(iB))-np.array([2, 3, 4])) > 1e-10):
    print('iB incorrect!\n')
