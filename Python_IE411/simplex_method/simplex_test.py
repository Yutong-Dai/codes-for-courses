# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 16:04:29 2017

@author: Siqi Miao
"""
# Test the feasibility and optimality of a basic vector for the
# linear program
#
# indices of iB, iN start with 1
import numpy as np
from numpy.linalg import norm

def simplex_test(A,b,c,iB,xB):

    iB = [i-1 for i in iB]
        
    [m,n] = A.shape
    eps = 1.0e-12
    
    X=[]
    eta=[]
    isfeasible=[]
    isoptimal=[]
    
    X = np.zeros((n,1))
    X[iB] = xB
    eta=c*X

    err = norm(A*X-b)
  
    isfeasible = 0
    if (err < eps) and min(X) >= -eps:
        isfeasible = 1
        
        
    temp = list(range(n))
    iN = []
    for each in temp:
        if each not in iB:
            iN.append(each)
    
    Cb = c[:,iB]  
    Cn = c[:,iN] 
    B = A[:,iB]
    N = A[:,iN]
    
    Binv = B.I
    
    ctilde = []
    for i in range(len(iN)):
        ctilde.append(Cn[:,i] - Cb * Binv * A[:,iN[i]])
    
    if not (min(ctilde) >= -eps):
        isoptimal = 0
        
    else:
        isoptimal = 1
        
    zN = ctilde


    return [X,eta,isfeasible,isoptimal,zN]