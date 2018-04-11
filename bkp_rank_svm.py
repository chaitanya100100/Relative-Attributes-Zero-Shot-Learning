import cvxpy as cvx

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

    # A = vstack((O, S))
    # C = np.concatenate((C_O, C_S))

    w = cvx.Variable(X.shape[1])
    epsilon = cvx.Variable(O.shape[0])
    gamma = cvx.Variable(S.shape[0])

    obj = cvx.Minimize( cvx.sum_squares(w) + cvx.sum_squares(epsilon) + cvx.sum_squares(gamma) )
    constraints = [
        (O*X)*w >= 1 - epsilon,
        cvx.abs((S*X)*w) <= gamma,
        gamma >= 0,
        epsilon >= 0
    ]

    print "start solving"
    prob = cvx.Problem(obj,constraints)
    prob.solve()
    return w.value
