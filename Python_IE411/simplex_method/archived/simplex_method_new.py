import numpy as np
from simplex_step import simplex_step
from simplex_init import find_negative_index


def simplex_init_modified(A, b, c):
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
            istatus, iB, iN, xB, tableau = 4, None, None, None, None
            return istatus, iB, iN, xB
    iB = iB - 1
    optimal_cost = np.matmul(c_phase_I[0, iB].reshape(1, -2), xB)
    if optimal_cost > 0:
        istatus, iB, iN, xB, tableau = 16, None, None, None, None
        return istatus, iB, iN, xB
    if optimal_cost == 0:
        #print("optimal basis is found!")
        istatus = 0
        artificial_idx = np.arange(c.shape[1], c.shape[1] + b.shape[0])
        artificial_in_basis = np.intersect1d(artificial_idx, iB)
        tableau = np.matmul(Binv, A_new)
        #c_new = np.concatenate((c, np.zeros(A.shape[0]).reshape(1, -1)), axis=1)
        #reduced_cost = c - np.matmul(np.matmul(c_new[0, iB], Binv), A)
        if len(artificial_in_basis) == 0:
            #print("no artificial variable in the final basis")
            return istatus, iB+1, iN, xB, tableau[:, 0:A.shape[1]]
        else:
            #print("artificial variable in the final basis")
            for xl in artificial_in_basis:
                row_l = tableau[np.where(iB == xl), :c.shape[1]]
                if np.sum(row_l) == 0:
                    tableau = np.delete(tableau,  np.where(iB == xl), axis=0)
                    xB = np.delete(xB, np.where(iB == xl))
                    iB = np.delete(iB, np.where(iB == xl))
            iN = np.setdiff1d(range(c.shape[1]), iB)
            iB = iB + 1
            iN = iN + 1
            xB = xB.reshape(-1, 1)
            return istatus, iB, iN, xB, tableau[:, 0:A.shape[1]]


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
    init_status, iB, iN, xB, tableau = simplex_init_modified(A, b, c)
    if init_status == 4:
        istatus, X, eta, iB, iN, xB = 16, None, None, None, None, None
        return istatus, X, eta, iB, iN, xB
    elif init_status == 16:
        istatus, X, eta, iB, iN, xB = 4, None, None, None, None, None
        return istatus, X, eta, iB, iN, xB
    else:
        istatus_step = 1000
        while istatus_step != -1:
            istatus_step, iB, iN, xB, _ = simplex_step(tableau, xB, c, iB, iN, xB, irule=1)
            if istatus_step == 16:
                istatus, X, iB, iN, xB = 32, None, None, None, None
                return istatus, X, eta, iB, iN, xB

        istatus = 0
        iB = iB - 1
        X = np.zeros((A.shape[1], 1))
        X[iB] = xB
        eta = np.matmul(c, X)
        iB = iB + 1
        return istatus, X, eta, iB, iN, xB


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
    # p = list(range(0, 6))
    # random.shuffle(p)
    # A = A[:, p]

    c = np.matrix([[-2, 1, 1, -1, -1, -1]], dtype=np.float64)
    #c = c[:, p]

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
    # print("Test11.py")
    # A1 = np.matrix([[-1,    1,     2],
    #                 [-1,    1,     1],
    #                 [0,    1,     1]], dtype=np.float64)

    # A = np.hstack((np.eye(3), A1))

    # b = np.matrix([[1],
    #                [2],
    #                [3]], dtype=np.float64)

    # iB = [1, 2, 3]
    # iN = [4, 5, 6]
    # xB = np.matrix(np.copy(b))
    # c = np.matrix([[0, 0, 0, -1, 2, 1]], dtype=np.float64)

    # # form an invertible matrix B and modify the problem
    # B = np.matrix([[4, 1, 0],
    #                [1, -2, -1],
    #                [1, 2, 4]], dtype=np.float64)
    # A = B*A
    # b = B*b

    # # modify c
    # N = A[:, [index_N-1 for index_N in iN]]
    # c1 = np.matrix([[1, 1, 0]], dtype=np.float64)
    # c2 = c[:, (4-1):6]+c1*B.I*N
    # c = np.hstack((c1, c2))

    # irule = 0
    # [istatus, X, eta, iB, iN, xB] = simplex_method(A, b, c, irule)

    # if (istatus != 32):
    #     print('istatus is wrong\n')
