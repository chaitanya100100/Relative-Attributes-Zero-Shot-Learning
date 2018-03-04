import cvxopt

'''
n - nummber of images
X - n*d array -> for all n images d dimensional feature vector
O - Ordered attributes array
S - Similiarity attributes array
A - p*n matrix -> p is number of preference pairs (+1,-1 sparse matrix)
C_O - training error penalization for ordered pairs
C_S - training error penalization for similiarity pairs
C - training error penalization vector for each preference pair
'''
