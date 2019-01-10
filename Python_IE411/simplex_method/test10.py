# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 21:02:17 2017

@author: Siqi Miao
"""

# test10.py
#
# An infeasible input to simplex method.
#
# indices of iB, iN start with 1

import numpy as np
from simplex_method import simplex_method


# first form an invertible matrix
R = np.matrix([[ 4, 1,  1],
               [ 1, 2,  1],
               [ 1, 1,  1]],dtype = np.float64)

# form a vector b which is in the span of R
b=R*np.matrix([[ 1],
               [-4],
               [-1]],dtype = np.float64)


B=np.matrix([[1, 1, 1],
             [1, 1, 0],
             [1, 0, 0]],dtype = np.float64)

A = np.hstack((R,B))

c=np.matrix([[-2, 1, 1, -1, -1, -1]],dtype = np.float64)



irule = 1
[istatus,X,eta,iB,iN,xB] = simplex_method(A,b,c,irule)


if (istatus !=4):
   print('istatus is wrong\n');
