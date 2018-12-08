from numpy.linalg import inv
import numpy as np


def simplex_step(A, b, c, iB, iN, xB, irule=0, Binv=""):
    """
    Take a single simplex method step for the linear program

      min:    c*x
      ST:     Ax=b
              x>=0,

    where A is an (m,n) matrix.

    That is, given a basic feasible vector vector described by the variables
    iB,iN,XB,

    Input Parameters:

        A - (n,m) constraint matrix
        b - (m,1) POSITIVE vector appearing in the constraint equation above
        c - (1,n) vector giving the coefficients of the objective function

        iB - (1,m) integer vector specifying the indices of the basic
            variables at the beginning of the simplex step
        iN - (1,n-m) integer vector specying the indices of the nonbasic
            variables at the beginning of the simplex step
        xB - (m,1) vector specifying the values of the basic
            variables at the beginning of the simplex step

        irule - integer parameter speciying which pivot rule to use:
            irule = 0 indicates that the smallest coefficient rule should be
                used
            irule = 1 indicates that Bland's rule should be used

    Output Parameters:

        istatus - integer parameter reporting the condition of the
            istatus = 0  indicates normal simplex method step
            completed
            istatus = 16 indicates the program is unbounded
            istatus = -1 indicates an optimal feasible vector has been
            found

        iB - integer vector specifying the m indices of the basic variables
            after the simplex step
        iN - integer vector specifying the n-m indices of the nonbasic
            variables after the simplex step
        xB - vector of length m specifying the values of the basic
            variables after the simplex step
    """
    if not isinstance(iB, np.ndarray):
        iB = np.array(iB)
    if not isinstance(iN, np.ndarray):
        iN = np.array(iN)
    if c.shape[1] > 1:
        c = c.T  # make c a column vector
    iB = iB - 1  # iB begin with index 1
    if isinstance(Binv, str):
        Binv = inv(A[:, iB])
        w_t = np.matmul(c[iB, 0].T, Binv)
    else:
        w_t = np.matmul(c[iB, 0].T, Binv)
    reduced_cost = c.T - np.matmul(w_t, A)
    if np.sum(reduced_cost >= 0) < reduced_cost.shape[1]:
        #print("not optimal yet")
        if irule == 0:
            j = np.argmin(reduced_cost)
        elif irule == 1:
            reduced_cost = np.squeeze(np.asarray(reduced_cost))
            for idx, flag in enumerate((reduced_cost < 0)):
                if flag:
                    j = idx
                    break
        else:
            raise ValueError("incorrect irule!")
        u = np.matmul(Binv, A[:, j])
        if np.sum(u > 0) == 0:
            #print("optimal cost is negative infinity")
            istatus = 16
            return [istatus, iB, iN, xB, Binv]
        else:
            # minimal ratio test
            u = np.asarray(u).reshape(-1,)
            idx = np.asarray(np.argwhere(u > 0)).reshape(-1)
            delta = np.asarray(xB).reshape(-1)[idx] / u[idx]
            l = np.argmin(delta)
            delta_ = delta[l]
            for i in range(A.shape[0]):
                if i != l:
                    xB[i, 0] -= delta_*u[i]
                else:
                    xB[i, 0] = delta_
            #print("an iteration finished")
            istatus = 0
            iB[l] = j
            Binv = inv(A[:, iB])
            iB = iB + 1  # recover the index of basic variables
            iN = np.setdiff1d(range(1, A.shape[1]+1), iB)
            return [istatus, iB, iN, xB, Binv]
    else:
        #print("reach optimal")
        istatus = -1
        return [istatus, iB, iN, xB, Binv]


if __name__ == "__main__":
    from numpy.linalg import norm
    print("My Test Case:")
    A1 = np.matrix([[1, 2,  2], [2, 1, 2], [2, 2,  1]], dtype=np.float64)
    A = np.hstack((A1, np.eye(3)))
    b = np.matrix([[20], [20], [20]], dtype=np.float64)
    iN = [1, 2, 3]
    iB = [4, 5, 6]
    xB = np.matrix(np.copy(b))
    c = np.matrix([[-10, -12, -12, 0, 0, 0, ]], dtype=np.float64)
    irule = 0
    [istatus, iB, iN, xB, Binv] = simplex_step(A, b, c, iB, iN, xB, irule=0)
    print("Status:{} | Basic variable index:{} | Basic variable value:{} | Non-basic Variable:{} | B inverse: {}".format(istatus, iB, xB, iN, Binv))
    print("===============================")
    print("Test1.py")
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
    print("Passed!")
    print("===============================")
    print("Test2.py")
    A = np.matrix([[-4,    1,     0,    -3,    -3,    -5],
                   [1,    -2,    -1,    -2,    -2,     3],
                   [1,     2,     4,     7,     7,    -1]], dtype=np.float64)
    b = np.matrix([[-2],
                   [-6],
                   [17]], dtype=np.float64)

    iB = [1, 2, 3]
    iN = [4, 5, 6]
    xB = np.matrix([[1], [2], [3]], dtype=np.float64)
    c = np.matrix([[1, 1, 1, 2, 5, 1]], dtype=np.float64)

    # take a step
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
    print("Passed!")
    print("===============================")
    print("Test3.py")
    A1 = np.matrix([[1, 1, 2],
                [1, 1, 1],
                [1, 1, 1]],dtype = np.float64)

    A = np.hstack((np.eye(3), A1))

    b = np.matrix([[1],
                [2],
                [3]],dtype = np.float64)
                

    iB = [1,2,3]
    iN = [4,5,6]
    xB = np.matrix(np.copy(b))
    c  = np.matrix([[0,0,0,-1,-2,1]],dtype = np.float64)

    # form an invertible matrix B and modify the problem
    B=np.matrix([[4, 1, 0],
                [1, -2, -1],
                [1, 2, 4]],dtype = np.float64)
    A=B*A
    b=B*b
    # modify c

    N=A[:,[index_N-1 for index_N in iN]]
    c1=np.matrix([1, 1, 1],dtype = np.float64)
    c2=c1*B.I*N+c[:,[index_N-1 for index_N in iN]]


    # take a step with the first rule
    irule = 0
    [istatus,iB,iN,x,Binv] = simplex_step(A,b,c,iB,iN,xB,irule)


    X = np.zeros((6,1),dtype = np.float64)
    X[[(b-1) for b in iB]] = xB

    if (istatus != 0):
        print('INCORRECT ISTATUS!\n')
    
    if (norm(X-np.matrix([[0],[1],[2],[0],[1],[0]]) > 1e-10)):
        print('INCORRECT STEP!\n')


    if (norm(np.array(sorted(iN))-np.array([1, 4, 6])) > 1e-10):
        print('iN incorrect!\n')


    if (norm(np.array(sorted(iB))-np.array([2, 3, 5])) > 1e-10):
        print('iB incorrect!\n')
    
    print("Passed!")
    print("===============================")
    print("Test4.py")
    A1 = np.matrix([[1,    1,     2],
                [1,    1,     1],   
                [1,    1,     1]],dtype = np.float64)   

    A = np.hstack((np.eye(3), A1))


    b = np.matrix([[1],
                [2],
                [3]],dtype = np.float64)
                

    iB = [1,2,3]
    iN = [4,5,6]
    xB = np.matrix(np.copy(b))
    c  = np.matrix([[0,0,0,-1,-2,1]],dtype = np.float64)
    # form an invertible matrix B and modify the problem
    B=np.matrix([[4, 1, 0],
                [1, -2, -1],
                [1, 2, 4]],dtype = np.float64)
    A=B*A
    b=B*b
    # modify c
    N=A[:,[index_N-1 for index_N in iN]]
    c1=np.matrix([[1, 1, 1]],dtype = np.float64)
    c2=c1*B.I*N+c[:,[index_N-1 for index_N in iN]];
    # take a step with Bland's rule
    irule = 1
    [istatus,iB,iN,xB,Binv] = simplex_step(A,b,c,iB,iN,xB,irule=1)

    X = np.zeros((6,1),dtype = np.float64)
    X[[(b-1) for b in iB]] = xB

    if (istatus != 0):
        print('INCORRECT ISTATUS!\n')
    
    if (norm(X-np.matrix([[0],[1],[2],[1],[0],[0]]) > 1e-10)):
        print('INCORRECT STEP!\n')


    if (norm(np.array(sorted(iN))-np.array([1, 5, 6])) > 1e-10):
        print('iN incorrect!\n')


    if (norm(np.array(sorted(iB))-np.array([2, 3, 4])) > 1e-10):
        print('iB incorrect!\n')
    print("Passed!")

    print("===============================")
    print("Test5.py")
    A1 = np.matrix([[-1,    1,     2],
                [-1,    1,     1],   
                [ 0,    1,     1]],dtype = np.float64)   

    A = np.hstack((np.eye(3), A1))
    b = np.matrix([[1],
                [2],
                [3]],dtype = np.float64)
    iB = [1,2,3]
    iN = [4,5,6]
    xB = np.matrix(np.copy(b))
    c  = np.matrix([[0,0,0,-1,2,1]],dtype = np.float64)
    # form an invertible matrix B and modify the problem
    B=np.matrix([[4, 1, 0],
                [1, -2, -1],
                [1, 2, 4]],dtype = np.float64)
    A=B*A
    b=B*b
    # modify c
    N=A[:,[index_N-1 for index_N in iN]]
    c1=np.matrix([[1, 1, 0]],dtype = np.float64)
    c2=c[:,(4-1):6]+c1*B.I*N
    c = np.hstack((c1,c2))
    # take a step.
    irule = 0
    [istatus,iB,iN,xB,Binv] = simplex_step(A,b,c,iB,iN,xB,irule)
    X = np.zeros((6,1),dtype = np.float64)
    X[[(b-1) for b in iB]] = xB
    if (istatus != 16):
        print('INCORRECT ISTATUS!\n')
    print("Passed!")

