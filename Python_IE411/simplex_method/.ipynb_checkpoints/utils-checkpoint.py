from numpy.linalg import inv
import numpy as np
def simplex_step(A, b, c, iB, iN, xB, Binv="", irule=0):
    if not isinstance(iB, np.ndarray):
        iB = np.array(iB)
    if not isinstance(iN, np.ndarray):
        iN = np.array(iN)
    if c.shape[1] > 1:
        c = c.T # make c a column vector
    # iB begin with index 1
    iB = iB - 1
    if isinstance(Binv, str):
        Binv = inv(A[:,iB])
        w_t = np.matmul(c[iB,0].T, Binv)
    else:
        w_t = np.matmul(c[iB,0].T, Binv)
    reduced_cost = c.T - np.matmul(w_t, A)
    if np.sum(reduced_cost>=0) < reduced_cost.shape[1]:
        print("not optimal yet")
        if irule == 0:
            j = 0#np.argmin(reduced_cost)
        u = np.matmul(Binv, A[:,j])
        if np.sum(u > 0) == 0:
            print("optimal cost is negative infinity")
            istatus = 16
            return [istatus]
        else:
            # minimal ratio test
            u = np.asarray(u).reshape(-1,)
            idx = np.asarray(np.argwhere(u>0)).reshape(-1)
            delta = np.asarray(xB).reshape(-1)[idx] / u[idx]
            l = np.argmin(delta)
            for i in range(A.shape[0]):
                if i != l:
                    xB[i,0] += xB[l,0] * (-A[i,j]/A[l,j]) 
            xB[l,0] /= A[l,j] 
            print("an iteration finished")
            istatus = -1
            iB[l] = j
            Binv = inv(A[:,iB])
            iB = iB + 1 # recover the index of basic variables
            iN = np.setdiff1d(range(1, A.shape[1]+1), iB)
            return [istatus, iB, iN, xB, Binv]
    else:
        print("reach optimal")
        istatus = 0
        return [istatus, iB, iN, xB, Binv]