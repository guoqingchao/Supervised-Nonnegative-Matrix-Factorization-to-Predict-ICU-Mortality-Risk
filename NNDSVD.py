#/usr/bin/env python
#coding=utf-8
import numpy as np
from scipy.linalg import svd
import copy
from numpy.linalg import norm



# This function NNDSVD implements the NNDSVD algorithm described in [1] for
# initialization of Nonnegative Matrix Factorization Algorithms.
#
# [W,H] = NNDSVD(A,k);
#
# INPUT
# ------------
#
# A    : the input nonnegative m x n matrix A
# k    : the rank of the computed factors W,H
# OUTPUT
# -------------
# W   : nonnegative m x k matrix
# H   : nonnegative k x n matrix
# References:
#
# [1] C. Boutsidis and E. Gallopoulos, SVD-based initialization: A head
#     start for nonnegative matrix factorization, Pattern Recognition,
#     Elsevier

#This function sets to zero the negative elements of a matrix
def pos(A):
    A[np.where(A < 0)] = 0
    return A

#This functions sets to zero the positive elements of a matrix and takes
#the absolute value of the negative elements

def neg(A):
    A[np.where(A > 0)] = 0
    return -A

def NNDSVD(A, k):
    if len(A[np.where(A < 0)]) > 0:
        print('the input matrix contains negative elements!')
    m, n = A.shape

    W = np.zeros((m, k))
    H = np.zeros((k, n))

    tmp = svd(A)
    U = tmp[0][:, 0: k + 1]
    S = tmp[1][0: k + 1]
    V = tmp[2][0: k + 1,:]
    S = np.diag(S)

    W[:, 0] = np.sqrt(S[0, 0]) * abs(U[:, 0])
    H[0, :] = np.sqrt(S[0, 0]) * abs((V[0, :]))

    i_lst = range(2,k + 1,1)
    for i in i_lst:
        uu = copy.deepcopy(U[:, i-1])
        vv = copy.deepcopy(V[i-1, :])
        uu1 = copy.deepcopy(U[:, i-1])
        vv1 = copy.deepcopy(V[i-1, :])
        uup = pos(uu)
        uun = neg(uu1)
        vvp = pos(vv)
        vvn = neg(vv1)
        n_uup = norm(uup)
        n_vvp = norm(vvp)
        n_uun = norm(uun)
        n_vvn = norm(vvn)
        termp = n_uup * n_vvp
        termn = n_uun * n_vvn
        if (termp >= termn):
            W[:, i-1] = np.sqrt(S[i-1, i-1] * termp) * uup / n_uup
            H[i-1, :] = np.sqrt(S[i-1, i-1] * termp) * vvp.T / n_vvp
        else:
            W[:, i-1] = np.sqrt(S[i-1, i-1] * termn) * uun / n_uun
            H[i-1, :] = np.sqrt(S[i-1, i-1] * termn) * vvn.T / n_vvn
    W[np.where(W < 0.0000000001)] = 0.1;
    H[np.where(H < 0.0000000001)] = 0.1;
    return (W, H)
