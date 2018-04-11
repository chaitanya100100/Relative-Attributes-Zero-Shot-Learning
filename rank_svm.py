import numpy as np
import scipy
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


def obj_fun_linear(w, C, out, X, A, n0):
    out[0:n0] = np.maximum(out[0:n0], np.zeros([n0, 1]))
    obj = np.sum(np.multiply(C, np.multiply(out, out))) / 2.0 + np.dot(w, w.T) / 2.0
    grad = w - (np.multiply(C, out).T * A * X).T
    sv = scipy.vstack(( out[0:n0] > 0, abs(out[n0:]) > 0 ))
    return obj, grad, sv

def rank_svm(X_, S_, O_, C_S, C_O):

    # opt
    max_itr = 10

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

        obj, grad, sv = obj_fun_linear(w, C, out, X, A, n0)
        
