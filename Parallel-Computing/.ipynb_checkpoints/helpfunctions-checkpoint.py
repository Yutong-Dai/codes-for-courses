import numpy as np
import pandas as pd
# import multiprocessing as mp
# from multiprocessing import Pool, freeze_support
# from itertools import repeat
# import time

def softT(m, threshold, zeta):
    """
    Soft-Threshold for solving univariate lasso.
    """
    # m: float; m_j in the thesis
    # t: float; truncation threshold
    # zeta: float; penalty factor, the same as the `lambda` in the thesis.
    if (zeta < abs(m)):
        z = (-m - zeta) / (2 * threshold) if (m < 0) else (-m + zeta) / (2 * threshold)
    else:
        z = 0
    return z

# def unpack_args(X, Y, beta, beta0, zeta, chunks = 4):
#     args_iterator = []
#     n, _ = X.shape
#     chunk_size = np.int(n/chunks) + 1
#     for i in range(chunks):
#         args_iterator.append((X[i*chunk_size:(i+1)*chunk_size], Y[i*chunk_size:(i+1)*chunk_size], beta, beta0, zeta))
#     return args_iterator



def calculate_probability(X, parameters):
    """
    Calculate the row-wise probability: 
        exp(beta_0 + X[i,].T * beta) / {1 + exp(beta_0 + X[i,].T * beta)}
    @Input:
        X: dataframe; n * p
        parameters: dictionary; {beta0: float, beta: 2-d np.array}
    @Output:
       prob_all: 2-d np.array; n*1; vectors containing row-wise probability
    """
    X_by_beta_plus_beta0 = np.matmul(X, parameters["beta"]) + parameters["beta0"]
    prob_all = X_by_beta_plus_beta0 / (1 + X_by_beta_plus_beta0)
    return (prob_all)

def calculate_lasso_solution(X, Y, prob_all, coordinate, parameters, threshold, zeta):
    """
    Calculate the univariate lasso solution at the `coordinate`.
    @Input:
        X: dataframe; n * p
        Y: dataframe; n * 1
        prob_all: np.array; n*1
        coordinate: int; range from 0 to p
        parameters: dictionary; {beta0: float, beta: 2-d np.array}
        zeta: float
        threshold: float
    @Output:
        solution: float
    """   
    n, _ = X.shape
    if coordinate == 0:
        gradient = (-0.5 / n) * np.sum(Y - prob_all)
        m_coordinate = gradient - 2 * threshold * parameters["beta0"]
        return (-m_coordinate) / (2 * threshold)
    else:
        y_by_x_coordinate = np.multiply(Y, X.iloc[:, [coordinate-1]])
        prob_all_by_x_coordinate = np.multiply(prob_all, X.iloc[:, [coordinate-1]])
        gradient = (-0.5 / n) * np.sum(y_by_x_coordinate - prob_all_by_x_coordinate)
        m_coordinate = gradient - 2 * threshold * parameters['beta'][coordinate-1]
        return softT(m_coordinate, threshold, zeta)

def calculate_object_function(X, Y, parameters, zeta):
    """
    Calculate objetc function value.
    @Input:
        X: dataframe; n * p
        Y: dataframe; n * 1
        parameters: dictionary; {beta0: float, beta: 2-d np.array}
        zeta: float
    @Output:
        fval: 2-d np.array; n*1
    """
    X_by_beta_plus_beta0 = np.matmul(X, beta) + beta0
    log_likelihood = np.sum(Y * X_by_beta_plus_beta0 - np.log(1 + np.exp(X_by_beta_plus_beta0)))
    fval = (-log_likelihood) / (2 * n) + zeta * np.sum(np.abs(beta)) 
    return fval

if __name__ == "__main__":
    data = pd.read_csv("./toyexample.csv")
    data = data.drop(["Unnamed: 0"], axis=1)
    X = data.drop('y', axis=1)
    Y = data.iloc[:,-1].values.reshape(-1,1)
    n, p = X.shape
    # rand_seed = np.random.RandomState(1234)
    # n, p = 1000, 100
    # X = rand_seed.beta(1,2,size=(n, p))
    # Y = rand_seed.binomial(n=1, p=0.2, size=n).reshape(-1,1)
    beta = np.ones(p).reshape(-1,1)
    beta0 = 1
    zeta = 0.1
    # parallel = False
#     print("Shape of X is n={}, p={}".format(n,p))
#     serial_start = time.time()
#     serila_result = calculate_object_function(X, Y, beta, beta0, zeta)
#     serial_end = time.time()
#     print("Serial took: [{}] s".format(serial_end - serial_start))
#     if parallel:
#         num_cores = mp.cpu_count()
#         pool_start = time.time()
#         args = unpack_args(X, Y, beta, beta0, zeta, chunks=10)
#         with Pool(processes = num_cores) as pool:
#             pool_result = pool.starmap(calculate_object_function, args)
#         pool_end = time.time()
#         print("Pool took: [{}] s".format(pool_end - pool_start))
    
#     try:
#         print("Difference betwwen two results: {}".format((np.sum(pool_result) - np.sum(serila_result))/n))
#         print(tt-t)
#     except NameError:
#         pass