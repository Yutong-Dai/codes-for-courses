# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 21:02:17 2017

@author: Siqi Miao
"""

# test8.py
#
# Tough initialization.  This one should only work if
# you've done the extra credit.
#
# indices of iB, iN start with 1

import numpy as np
import random
from numpy.linalg import norm,cond
from simplex_init import simplex_init

eps=1.0e-10;

# first form an invertible matrix
R = np.matrix([[ 4, 1,  1],
               [-1, 2,  1],
               [ 1, 1, -1]],dtype = np.float64)

# form a vector b which is in the span of 2 vectors of R
b=R*np.matrix([[1],
               [2],
               [0]],dtype = np.float64)


B=np.matrix([[1, 1, 1],
             [1, 1, 0],
             [1, 0, 0]],dtype = np.float64)
A = np.hstack((R,B))


# form a random permutation
p = list(range(0,6))
random.shuffle(p) 
A=A[:,p]


# c doesn't matter for this test
c=np.matrix([[-1, -1, -1, -1, -1, -1]],dtype = np.float64)

[istatus,iB,iN,xB] = simplex_init(A,b,c)

if (istatus !=0) :
   print('looks like you did not do the extra credit!\n')


# test feasibility
X = np.zeros((6,1),dtype = np.float64)
X[[(b1-1) for b1 in iB]] = xB



if (norm(A*X-b) > eps):
   print('NOT FEASIBLE!!!\n');

if (min(X) < 0):
   print('NOT FEASIBLE!!!\n');


# test that we have a basis
if ((1/cond(A[:,[(b-1) for b in iB]],2)) > 1.0e6):
   print('NOT BASIC!!!\n');

