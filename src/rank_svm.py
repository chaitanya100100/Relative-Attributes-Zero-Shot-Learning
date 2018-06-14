import numpy as np
import scipy
from scipy.optimize import least_squares
import time

'''
n - nummber of images
X - n*d array -> for all n images d dimensional feature vector
O - Ordered attributes array
S - Similiarity attributes array
A - [O;S] p*n matrix -> p is number of preference pairs (+1,-1 sparse matrix)
C_O - training error penalization for ordered pairs
C_S - training error penalization for similiarity pairs
C - [C_O;C_S] -> training error penalization vector for each preference pair
w - weight vector to be learnt
'''

def obj_fun_linear(w, C, out):
    out[0:n0] = np.maximum(out[0:n0], np.zeros([n0, 1]))
    obj = np.sum(np.multiply(C, np.multiply(out, out))) / 2.0 + np.dot(w.T, w) / 2.0
    grad = w - (np.multiply(C, out).T * A * X).T
    sv = scipy.vstack(( out[0:n0] > 0, abs(out[n0:]) > 0 ))
    return obj[0, 0], grad, sv

def hess_vect_mult(w, sv, C, grad):
    w = np.matrix(w).T
    y = w
    z = np.multiply(np.multiply(C, sv), A * (X * w))
    y = y + ((z.T * A) * X).T + grad
    y = y.A1
    return y

def line_search_linear(w, d, out, C):
    t = 0
    Xd = A * (X * d)
    wd = w.T * d
    dd = d.T * d

    while True:

        out2 = out - t * Xd
        sv = np.nonzero( scipy.vstack(( out2[0:n0] > 0, abs(out2[n0:]) > 0 )) )[0]
        g = wd + t * dd - np.multiply(C[sv], out2[sv]).T * Xd[sv]
        h = dd + Xd[sv].T * np.multiply(Xd[sv], C[sv])
        g = g[0, 0]
        h = h[0, 0]
        t = t - g / h
        if g * g / h < 1e-8:
            break
    out = out2
    return t, out

def rank_svm(X_, S_, O_, C_S, C_O):

    # opt
    max_itr = 10
    prec = 1e-8
    cg_prec = 1e-8
    cg_max_itr = 20

    global X
    global A
    global n0

    X = X_; A = O_; B = S_;
    n0 = A.shape[0]
    d = X.shape[1]
    n = A.shape[0]

    w = scipy.matrix(scipy.zeros([d, 1]))

    itr = 0
    C = np.vstack((C_O, C_S))

    out = scipy.matrix(scipy.vstack( (scipy.ones([A.shape[0], 1]), scipy.zeros([B.shape[0], 1])) )) \
        - scipy.sparse.vstack((A, B)) * X * w

    A = scipy.sparse.vstack((A, B))


    while True:
        itr = itr + 1
        if itr > max_itr:
            print "Maximum number of Newton steps reached"
            break
        start_time = time.time()


        obj, grad, sv = obj_fun_linear(w, C, out)
        res = least_squares(hess_vect_mult, np.zeros(w.shape[0]), ftol = cg_prec, xtol = cg_prec, gtol = cg_prec, args = (sv, C, grad))
        step = np.matrix(res.x).T
        t, out = line_search_linear(w, step, out, C)
        w = w + t * step;

        check = - step.T * grad
        check = check[0, 0]
        print check, prec*obj
        if check < prec * obj:
            break

        end_time = time.time()
        print "Iteration", itr, "time elapesed", end_time - start_time

    print w
    return w
