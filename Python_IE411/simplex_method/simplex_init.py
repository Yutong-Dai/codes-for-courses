'''
File: simplex_init.py
Author: Yutong Dai (rothdyt@gmail.com)
File Created: Saturday, 2018-12-15 00:56
Last Modified: Saturday, 2018-12-15 00:57
--------------------------------------------
Desscription:
'''

import numpy as np
from simplex_step import simplex_step


def find_negative_index(b):
    row_idxs = np.argwhere(np.asarray(b).reshape(-1) < 0).reshape(-1)
    return row_idxs


def simplex_init(A, b, c):
    """
     Attempt to find a basic feasible vector for the linear program

         max:    c*x
         ST:     Ax=b
                 x>=0,

     where A is a (m,n) matrix.

    Input Parameters:

        A - (n,m) constraint matrix
        b - (m,1) vector appearing in the constraint equation above
        c - (1,n) vector giving the coefficients of the objective function


    Output Parameters:

        istatus - integer parameter reporting the condition of the
            istatus = 0  indicates a basic feasible vector was found
            istatus = 4  indicates that the initialization procedure failed
            istatus = 16  indicates that the problem is infeasible

        iB - integer vector of length m specifying the indices of the basic
            variables
        iN - integer vector of length n-m specying the indices of the nonbasic
            variables
        xB - vector of length m specifying the values of the basic
            variables
    """
    A_new, b_new = A, b
    A_new[find_negative_index(b)] = -A[find_negative_index(b)]
    b_new[find_negative_index(b)] = -b[find_negative_index(b)]
    A_new = np.hstack((A_new, np.eye(b.shape[0])))
    # problem setup
    c_phase_I = np.zeros(A_new.shape[1]).reshape(1, -1)
    c_phase_I[0, c.shape[1]:] = np.ones(b.shape[0])
    iB = np.arange(c.shape[1], c.shape[1] + b.shape[0]) + 1  # index begin with 1 for input
    iN = np.arange(0, c.shape[1]) + 1
    xB = np.matrix(np.copy(b))
    istatus_step = 1000
    while istatus_step != -1:
        try:
            istatus_step, iB, iN, xB, Binv = simplex_step(A_new, b_new, c_phase_I, iB, iN, xB, irule=0)
        except np.linalg.LinAlgError:
            raise ValueError("iB cannot form a basis!")
        if istatus_step == 16:
            istatus, iB, iN, xB = 4, None, None, None
            return istatus, iB, iN, xB
    iB = iB - 1
    optimal_cost = np.matmul(c_phase_I[0, iB].reshape(1, -2), xB)
    if optimal_cost > 0:
        istatus, iB, iN, xB = 16, None, None, None
        return istatus, iB, iN, xB
    if optimal_cost == 0:
        #print("optimal basis is found!")
        istatus = 0
        artificial_idx = np.arange(c.shape[1], c.shape[1] + b.shape[0])
        artificial_in_basis = np.intersect1d(artificial_idx, iB)
        if len(artificial_in_basis) == 0:
            #print("no artificial variable in the final basis")
            return istatus, iB+1, iN, xB
        else:
            tableau = np.matmul(Binv, A_new)
            for xl in artificial_in_basis:
                row_l = tableau[np.where(iB == xl), :c.shape[1]]
                if np.sum(row_l) == 0:
                    xB = np.delete(xB, np.where(iB == xl))
                    iB = np.delete(iB, np.where(iB == xl))
            iN = np.setdiff1d(range(c.shape[1]), iB)
            iB = iB + 1
            iN = iN + 1
            return istatus, iB, iN, xB
    # if optimal_cost == 0:
    #     istatus = 0
    #     return istatus, iB+1, iN, xB


if __name__ == "__main__":
    import numpy as np
    from numpy.random import randn
    import random
    from numpy.linalg import norm, cond
    from simplex_step import simplex_step
    print("===============================")
    print("Test6.py")
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
    print("Passed!")
    print("===============================")
    print("Test7.py")
    eps = 1.0e-10

    # first form an invertible matrix
    R = np.matrix([[4, 1,  1],
                   [-1, 2,  1],
                   [1, 1, -1]], dtype=np.float64)

    # form a vector b which is in the span of R
    b = R*abs(randn(3, 1))

    B = np.matrix([[1, 1, 1],
                   [1, 1, 0],
                   [1, 0, 0]], dtype=np.float64)
    A = np.hstack((R, B))

    # form a random permutation
    p = list(range(0, 6))
    random.shuffle(p)
    A = A[:, p]

    # c doesn't matter for this test
    c = np.matrix([[-1, -1, -1, -1, -1, -1]], dtype=np.float64)
    [istatus, iB, iN, xB] = simplex_init(A, b, c)

    X = np.zeros((6, 1), dtype=np.float64)
    X[[(b1-1) for b1 in iB]] = xB

    if (istatus != 0):
        print('istatus wrong!\n')

    # test feasibility
    if (norm(A*X-b) > eps):
        print('NOT FEASIBLE!!!\n')

    if (min(X) < 0):
        print('NOT FEASIBLE!!!\n')

    # test that we have a basis
    if (1/cond(A[:, [(b-1) for b in iB]])) > 1.0e6:
        print('NOT BASIC!!!\n')
    print("Passed!")
    print("===============================")
    print("Test8.py")
    ps = 1.0e-10

    # first form an invertible matrix
    R = np.matrix([[4, 1,  1],
                   [-1, 2,  1],
                   [1, 1, -1]], dtype=np.float64)

    # form a vector b which is in the span of 2 vectors of R
    b = R*np.matrix([[1],
                     [2],
                     [0]], dtype=np.float64)

    B = np.matrix([[1, 1, 1],
                   [1, 1, 0],
                   [1, 0, 0]], dtype=np.float64)
    A = np.hstack((R, B))

    # form a random permutation
    p = list(range(0, 6))
    random.shuffle(p)
    A = A[:, p]

    # c doesn't matter for this test
    c = np.matrix([[-1, -1, -1, -1, -1, -1]], dtype=np.float64)

    [istatus, iB, iN, xB] = simplex_init(A, b, c)

    if (istatus != 0):
        print('looks like you did not do the extra credit!\n')
    # test feasibility
    X = np.zeros((6, 1), dtype=np.float64)
    X[[(b1-1) for b1 in iB]] = xB

    if (norm(A*X-b) > eps):
        print('NOT FEASIBLE!!!\n')

    if (min(X) < 0):
        print('NOT FEASIBLE!!!\n')

    # test that we have a basis
    if ((1/cond(A[:, [(b-1) for b in iB]], 2)) > 1.0e6):
        print('NOT BASIC!!!\n')
    print("Passed!")
    print("===============================")
