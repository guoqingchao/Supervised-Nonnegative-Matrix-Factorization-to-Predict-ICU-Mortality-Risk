#/usr/bin/env python
#coding=utf-8
import copy
import time

import numpy as np
from numpy import exp, log
from numpy.linalg import norm


debug = 0

# This function is used to calculate the Lagrange function values
def objective_function(P, R, S, X, Y, U, V, W, b, lambda1, lambda2, lambda3, lambda4):
    UW = U.dot(W)
    WtW = W.T.dot(W)
    F = 0.5 * 1.0/(P * S)*norm(X - U.dot(V),'fro') ** 2 + float(lambda1)/P * log(1 + exp(- (UW + b) * Y)).sum(axis = 0)\
    + 0.5 * float(lambda2)/R * (WtW + b ** 2) \
    + 0.5 * float(lambda3)/(P * R) * norm(U,'fro') ** 2 + 0.5 * float(lambda4)/(R * S) * norm(V,'fro') ** 2
    return F

# This function is used to calculate the Lagrange function values related to variable U
def objective_U(P, R, S, X, Y, U, V, W, b, lambda1, lambda3):
    UW=U.dot(W)
    F = 0.5 * 1.0 / (P * S) * norm(X - U.dot(V),'fro') ** 2 + \
        float(lambda1) / P * log(1 + exp(-(UW + b) * Y)).sum(axis = 0) \
        + 0.5 * float(lambda3) / (P * R) * norm(U,'fro') ** 2
    return F

# This function is used to solve variable U with other variables fixed
def subprob_U(P, R, S, X, Y, Uinit, V, W, b, lambda1, lambda3, tol, maxiterU):
    maxiter = maxiterU
    U = copy.deepcopy(Uinit)
    VVt = V.dot(V.T)
    XVt = X.dot(V.T)
    YWt = Y.dot(W.T)
    alpha = 1.0
    beta = 0.1
    obj_U = []
    iter_lst = range(1, maxiter + 1, 1)
    ineriter_lst = range(1, 21, 1)
    for iter in iter_lst:
        UW = U.dot(W)
        Uwby = 1 + exp((UW + b) * Y)
        grad = 1.0 / (P * S) * (U.dot(VVt) - XVt) - float(lambda1) / P * (YWt / Uwby) + float(lambda3) / (P * R) * U
        projgrad = norm(np.matrix(grad[np.logical_or(grad < 0,U > 0)]),'fro')
        if debug:
            print ('Step into U loop\n')
            print ('%18.16f\n'%projgrad)
        if projgrad < tol:
            if debug:
                print('1')
                print ('%18.16f\n'%projgrad)
                print ('%18.16f\n'%tol)
            break
        objold = objective_U(P, R, S, X, Y, U, V, W, b, lambda1, lambda3)
        obj_U.append(objold)
        for ineriter in ineriter_lst:
            Un = U - alpha * grad
            Un[np.where(Un < 0)] = 0
            d = Un - U
            gradd = (grad * d).sum()
            objnew = objective_U(P, R, S, X, Y, Un, V, W, b, lambda1, lambda3)
            suff_decr = objnew - objold - 0.01 * gradd < 0
            if ineriter == 1:
                decr_alpha = not suff_decr
                Up = U
            if decr_alpha:
                if suff_decr:
                    U = Un
                    if debug:
                        print('2')
                    break
                else:
                    alpha = alpha * beta
            else:
                if not suff_decr or np.array_equal(Up, Un):
                    U=Up
                    if debug:
                        print('3')
                    break
                else:
                    alpha = alpha/beta
                    Up=Un
    if iter == maxiter:
        print('max iter in U\n')
    return U, grad, iter, obj_U

# This function is used to calculate the Lagrange function values related to variable W
def objective_W(P, R, S, Y, U, W, b, lambda1, lambda2):
    UW = U.dot(W)
    WtW = W.T.dot(W)
    F = float(lambda1) / P * log(1 + exp(- (UW + b) * Y)).sum() + 0.5 * float(lambda2) / R * (WtW)
    return F

# This function is used to solve variable W with other variables fixed
def subprob_W(P, R, S, Y, U, Winit, b, lambda1, lambda2, tol, maxiter):
    W = copy.deepcopy(Winit)
    alpha = 1.0
    beta = 0.1
    obj_W = []
    iter_lst = range(1, maxiter + 1, 1)
    ineriter_lst = range(1, 21, 1)
    for iter in iter_lst:
        UW = U.dot(W)
        Uwby = 1 + exp((UW + b) * Y)
        grad = float(lambda2) / R * W - np.reshape(float(lambda1) / P * ((U * Y) / Uwby).sum(axis = 0), (-1, 1))
        projgrad = norm(grad, 'fro')
        if debug:
            print ('Step into W loop\n')
            print ('%18.16f\n'%projgrad)
        if projgrad < tol:
            if debug:
                print('1')
                print ('%18.16f\n'%projgrad)
                print ('%18.16f\n'%tol)
            break
        objold = objective_W(P, R, S, Y, U, W, b, lambda1, lambda2)
        obj_W.append(objold)
        for ineriter in ineriter_lst:
            Wn = W - alpha * grad
            d = Wn - W
            gradd = (grad * d).sum()
            objnew = objective_W(P, R, S, Y, U, Wn, b, lambda1, lambda2)
            suff_decr = objnew - objold - 0.01 * gradd < 0
            if ineriter == 1:
                decr_alpha = not suff_decr
                Wp = W
            if decr_alpha:
                if suff_decr:
                    W = Wn
                    if debug:
                        print('2')
                    break
                else:
                    alpha = alpha * beta
            else:
                if not suff_decr or np.array_equal(Wp, Wn):
                    W = Wp
                    if debug:
                        print('3')
                    break
                else:
                    alpha = alpha / beta
                    Wp=Wn
    if iter == maxiter:
        print('max iter in W\n')
    return W,grad, iter, obj_W

# This function is used to calculate the Lagrange function values related to variable b
def objective_b(P, R, S, Y, U, W, b, lambda1, lambda2):
    UW = U.dot(W)
    F = float(lambda1) / P * log(1 + exp(-(UW + b) * Y)).sum() + 0.5 * float(lambda2) / R * (b ** 2)
    return F

# This function is used to solve variable b with other variables fixed
def subprob_b(P, R, S, Y, U, W, binit, lambda1, lambda2, tol, maxiter):
    b = copy.deepcopy(binit)
    UW = U.dot(W)
    alpha = 1.0
    beta = 0.1
    obj_b = []
    iter_lst = range(1, maxiter + 1, 1)
    ineriter_lst = range(1, 21, 1)
    for iter in iter_lst:
        grad = float(lambda2) / R * b - np.reshape(float(lambda1) / P * (Y / (1 + exp((UW + b) * Y))).sum(axis = 0),(-1,1))
        projgrad = norm(grad,'fro')
        if debug:
            print ('Step into b loop\n')
            print ('%18.16f\n'%projgrad)
        if projgrad < tol:
            if debug:
                print('1')
                print ('%18.16f\n'%projgrad)
                print ('%18.16f\n'%tol)
            break
        objold = objective_b(P, R, S, Y, U, W, b, lambda1, lambda2)
        obj_b.append(objold)
        for ineriter in ineriter_lst:
            bn = b - alpha * grad
            d = bn - b
            gradd = (grad * d).sum()
            objnew = objective_b(P, R, S, Y, U, W, bn, lambda1, lambda2)
            suff_decr = objnew - objold - 0.01 * gradd < 0
            if ineriter == 1:
                decr_alpha = not suff_decr
                bp = b
            if decr_alpha:
                if suff_decr:
                    b = bn
                    if debug:
                        print('2')
                    break
                else:
                    alpha = alpha * beta
            else:
                if not suff_decr or np.array_equal(bp, bn):
                    b = bp
                    if debug:
                        print('3')
                    break
                else:
                    alpha = alpha / beta
                    bp = bn
    if iter == maxiter:
        print('max iter in b\n')
    return b, grad, iter, obj_b

# This function is used to calculate the Lagrange function values related to variable V
def objective_V(P,R,S,X,U,V,lambda4):
    F = 0.5 * 1.0 / (P * S) * norm(X - U.dot(V),'fro') ** 2 + 0.5 * float(lambda4) / (R * S) * norm(V,'fro') ** 2
    return F

# This function is used to solve variable V with other variables fixed
def subprob_V(P, R, S, X, U, Vinit, lambda4, tol, maxiter):
    V = copy.deepcopy(Vinit)
    UtX = U.T.dot(X)
    UtU = U.T.dot(U)
    alpha = 1.0
    beta = 0.1
    obj_V=[]
    iter_lst = range(1, maxiter + 1, 1)
    ineriter_lst = range(1, 21, 1)
    for iter in iter_lst:
        grad = 1.0 / (P * S) * (UtU.dot(V) - UtX) + float(lambda4) / (R * S) * V
        projgrad = norm(np.matrix(grad[np.logical_or(grad < 0, V > 0)]), 'fro')
        if debug:
            print ('Step into V loop\n')
            print ('%18.16f\n' %projgrad)
        if projgrad < tol:
            if debug:
                print('1')
                print ('%18.16f\n' %projgrad)
                print ('%18.16f\n' %tol)
            break
        objold = objective_V(P, R, S, X, U, V, lambda4)
        obj_V.append(objold)
        for ineriter in ineriter_lst:
            Vn = V-alpha * grad
            Vn[np.where(Vn < 0)] = 0
            d = Vn - V
            gradd = (grad * d).sum()
            objnew = objective_V(P, R, S, X, U, Vn, lambda4)
            suff_decr = objnew - objold - 0.01 * gradd < 0
            if ineriter == 1:
                decr_alpha = not suff_decr
                Vp = V
            if decr_alpha:
                if suff_decr:
                    V = Vn
                    if debug:
                        print('2')
                    break
                else:
                    alpha = alpha*beta
            else:
                if not suff_decr or np.array_equal(Vp, Vn):
                    V = Vp
                    if debug:
                        print('3')
                    break
                else:
                    alpha = alpha / beta
                    Vp = Vn
    if iter == maxiter:
        print('max iter in V\n')
    return V, grad, iter, obj_V

# This function snmfpg is used to solve the supervised nonnegative matrix factorization (SNMF) with projected gradient
def snmfpg(X, Y, Uinit, Vinit, Winit, binit, lambda1, lambda2, lambda3, lambda4, tol, timelimit, maxiter):
    U = Uinit
    V = Vinit
    W = Winit
    b = binit
    initt = time.time()
    P = U.shape[0]
    R = V.shape[0]
    S = V.shape[1]
    VVt = V.dot(V.T)
    XVt = X.dot(V.T)
    YWt = Y.dot(W.T)
    UW = U.dot(W)
    UtU = U.T.dot(U)
    Uwby = 1 + exp((UW + b) * Y)
    gradU = 1.0 / (P * S) * (U.dot(VVt) - XVt) - float(lambda1) / P * (YWt / Uwby) + float(lambda3) / (P * R) * U
    gradV = 1.0 / (P * S) * (UtU.dot(V) - U.T.dot(X)) + float(lambda4) / (R * S) * V
    gradW = float(lambda2) / R * W - np.reshape(float(lambda1) / P * ((U * Y) / Uwby).sum(axis = 0),(-1,1))
    gradb = float(lambda2) / R * b - np.reshape(float(lambda1) / P * (Y / Uwby).sum(axis = 0),(-1,1))
    initgrad = norm(gradU,'fro') + norm(gradV,'fro') + norm(gradW,'fro') + norm(gradb,'fro')
    if debug:
        print('Init gradient norm:')
        print ('%18.16f\n'%initgrad)
    tolU = max(1e-10,tol) * initgrad
    tolV = tolU
    tolW = tolV
    tolb = tolW

    maxiter_sub = 50;
    obj_total_lst = []
    for iter in range(1, maxiter + 1, 1):
        if debug:
            print('Iteration:')
            print ('%f\n'%iter)
        ngradU = norm(np.matrix(gradU[np.logical_or(gradU < 0,U > 0)]),'fro');
        ngradV = norm(np.matrix(gradV[np.logical_or(gradV < 0,V > 0)]),'fro');
        ngradW = norm(gradW,'fro');
        ngradb = norm(gradb,'fro');
        projnorm = ngradU + ngradV + ngradW + ngradb;
        loss_nmf = 0.5 * 1.0 / (P * S) * norm(X - U.dot(V),'fro') ** 2
        loss_logit = float(lambda1) / P * (log(1 + exp(-(U.dot(W) + b) * Y))).sum(axis = 0)
        loss_wb = 0.5 * float(lambda2) / R * (W.T.dot(W) + b ** 2)
        loss_U = 0.5 * float(lambda3) / (P * R) * norm(U,'fro') ** 2
        loss_V = 0.5 * float(lambda4) / (R * S) * norm(V,'fro') ** 2
        if debug:
            print('--------------------')
            projnormtol = tol * initgrad
            print ('%18.16f\n'%projnorm)
            print ('%18.16f\n'%projnormtol)
            timelimitdiff = time.time() - initt
            print ('%18.16f\n'%timelimitdiff)
            print ('%18.16f\n'%timelimit)
        if projnorm < tol * initgrad or time.time() - initt > timelimit:
            if debug:
                print('******************************************************')
                projnormtol = tol * initgrad
                print ('%18.16f\n'%projnorm)
                print ('%18.16f\n'%projnormtol)
                timelimitdiff = time.time() - initt
                print ('%18.16f\n'%timelimitdiff)
                print ('%18.16f\n'%timelimit)
            break

        U,gradU,iterU,obj_U = subprob_U(P, R, S, X, Y, U, V, W, b, lambda1, lambda3, tolU, maxiter_sub)
        if iterU == 1:
            tolU = 0.1 * tolU
        W,gradW,iterW,obj_W = subprob_W(P, R, S, Y, U, W, b, lambda1, lambda2, tolW, maxiter_sub)
        if iterW == 1:
            tolW = 0.1 * tolW
        b,gradb,iterb,obj_b = subprob_b(P, R, S, Y, U, W, b, lambda1, lambda2, tolb, maxiter_sub)
        if iterb == 1:
            tolb = 0.1 * tolb
        V,gradV,iterV,obj_V = subprob_V(P, R, S, X, U, V, lambda4, tolV, maxiter_sub)
        if iterV == 1:
            tolV = 0.1 * tolV
# The following five items are the loss values of the five items of snmfpg
        loss_nmf = 0.5 * 1.0 / (P * S) * norm(X - U.dot(V),'fro') ** 2
        loss_logit = float(lambda1) / P * (log(1 + exp(-(U.dot(W) + b) * Y))).sum(axis = 0)
        loss_wb = 0.5 * float(lambda2) / R * (W.T.dot(W) + b ** 2)
        loss_U = 0.5 * float(lambda3) / (P * R) * norm(U,'fro') ** 2
        loss_V = 0.5 * float(lambda4) / (R * S) * norm(V,'fro') ** 2
# obj_total_lst is used to see the convergence of outer loops
        tmp=objective_function(P, R, S, X, Y, U, V, W, b, lambda1, lambda2, lambda3, lambda4)
        obj_total_lst.append(tmp)
    return(U, V, W, b, loss_nmf, loss_logit, loss_wb, loss_U, loss_V, obj_total_lst, obj_U, obj_W, obj_b, obj_V)
