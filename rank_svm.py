from cvxopt import normal, uniform
from cvxopt.modeling import variable, dot, op, sum
import numpy as np

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

def rank_svm(X, S, O, C_S, C_O):
    A = np.concatenate(O, S)
    C = np.concatenate(C_O, C_S)

    w = variable(X.shape[1]) #X.shape[1] = d
    constraint1 = [np.ones([O.shape[0],1])] - (O*X)*w
    constraint2 = (S*X)*w - [zeros([S.shape[0],1])]
    constraints = np.concatenate(constraint1, constraint2)
    obj = (np.transpose(w).dot(w))/2 + sum(np.multiply( np.multiply(C, constraints), constraints ) ) #minimize this function

    epsilon = variable(constraint1.shape[0])
    gamma = variable(constraint2.shape[0])

    opt = op(obj,[constraint1 <= epsilon, constraint2 <= gamma, constraint2 >= -gamma])
    opt.solve()
    return opt, w
