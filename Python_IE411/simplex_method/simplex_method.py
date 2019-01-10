'''
File: simplex_method.py
Author: Yutong Dai (rothdyt@gmail.com)
File Created: Saturday, 2018-12-15 15:17
Last Modified: Saturday, 2018-12-15 15:35
--------------------------------------------
Desscription:
'''
import numpy as np
from simplex_step import simplex_step
from simplex_init import simplex_init


def simplex_method(A, b, c, irule):
    """
    Find a basic optimal solution for the linear program

     min:    c*x
     ST:     Ax=b
             x>=0,

    where A is an (m,n) matrix.

    Input Parameters:

    A - (n,m) constraint matrix
    b - (m,1) POSITIVE vector appearing in the constraint equation above
    c - (1,n) vector giving the coefficients of the objective function

    irule - integer parameter speciying which pivot rule to use:
        irule = 0 indicates that the smallest coefficient rule should be 
            used
        irule = 1 indicates that Bland's rule should be used

    Output Parameters:

    istatus - integer parameter reporting the condition of the 
        istatus = 0  indicates normal completeion (i.e., a solution
        has been found and reported)
        istatus = 4  indicates the program is infeasible
        istatus = 16 indicates the program is feasible but our initialization
        procedure has failed
        istatus = 32 indicates that the program is unbounded

    X  - vector of length n specifying the solution
    eta - the minimum value of the objective function 
    iB - integer vector specifying the m indices of the basic variables
        after the simplex step
    iN - integer vector specifying the n-m indices of the nonbasic 
        variables after the simplex step
    xB - vector of length m specifying the values of the basic
        variables after the simplex step

    """
    #init_status, iB, iN, xB, tableau = simplex_init_modified(A, b, c)
    init_status, iB, iN, xB = simplex_init(A, b, c)
    if init_status == 4:
        istatus, X, eta, iB, iN, xB = 16, None, None, None, None, None
        return istatus, X, eta, iB, iN, xB
    elif init_status == 16:
        istatus, X, eta, iB, iN, xB = 4, None, None, None, None, None
        return istatus, X, eta, iB, iN, xB
    else:
        istatus_step = 1000
        count = 0
        while istatus_step != -1:
            istatus_step, iB, iN, xB, _ = simplex_step(A, b, c, iB, iN, xB, irule=0, Binv="")
            count += 1
            if istatus_step == 16 or count > 100:
                istatus, X, eta, iB, iN, xB = 32, None, None, None, None, None
                return istatus, X, eta, iB, iN, xB

        istatus = 0
        iB = iB - 1
        eta = np.matmul(c[0, iB].reshape(1, -2), xB)
        X = np.zeros((A.shape[1], 1))
        X[iB] = xB
        return istatus, X, eta, iB+1, iN, xB


if __name__ == "__main__":
    import numpy as np
    import random
    from simplex_method import simplex_method
    from simplex_test import simplex_test
    print("===============================")
    print("Test9.py")
    # first form an invertible matrix
    R = np.matrix([[4, 1,  1],
                   [-1, 2,  1],
                   [1, 1, -1]], dtype=np.float64)

    # form a vector b which is in the span of R
    b = R*np.matrix([[1],
                     [2],
                     [1]], dtype=np.float64)

    B = np.matrix([[1, 1, 1],
                   [1, 1, 0],
                   [1, 0, 0]], dtype=np.float64)
    A = np.hstack((R, B))

    # form a random permutation
    p = list(range(0, 6))
    random.shuffle(p)
    A = A[:, p]

    c = np.matrix([[-2, 1, 1, -1, -1, -1]], dtype=np.float64)
    c = c[:, p]

    # test
    irule = 0
    [istatus, X, eta, iB, iN, xB] = simplex_method(A, b, c, irule)
    print(X, eta, iB, iN, xB)
    # return

    if (istatus != 0):
        print('istatus is wrong\n')

    [X, eta, isfeasible, isoptimal, zN] = simplex_test(A, b, c, iB, xB)

    if (isfeasible != 1):
        print('your solution is not feasible!!!\n')

    if (isoptimal != 1):
        print('your solution is not optimal!!!\n')

    print("===============================")
    print("Test10.py")
    R = np.matrix([[4, 1,  1],
                   [1, 2,  1],
                   [1, 1,  1]], dtype=np.float64)

    # form a vector b which is in the span of R
    b = R*np.matrix([[1],
                     [-4],
                     [-1]], dtype=np.float64)

    B = np.matrix([[1, 1, 1],
                   [1, 1, 0],
                   [1, 0, 0]], dtype=np.float64)

    A = np.hstack((R, B))

    c = np.matrix([[-2, 1, 1, -1, -1, -1]], dtype=np.float64)

    irule = 1
    [istatus, X, eta, iB, iN, xB] = simplex_method(A, b, c, irule)

    if (istatus != 4):
        print('istatus is wrong\n')
    print("===============================")
    print("Test11.py")
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

    irule = 0
    [istatus, X, eta, iB, iN, xB] = simplex_method(A, b, c, irule)

    if (istatus != 32):
        print('istatus is wrong\n')
