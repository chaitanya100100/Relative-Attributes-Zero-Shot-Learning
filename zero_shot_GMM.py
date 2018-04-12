import numpy as np
import scipy
from zero_shot_config import *
from sklearn.mixture import GaussianMixture as GMM
from scipy.stats import multivariate_normal
from numpy.linalg import inv, det

datadict = np.load('datadict.npy').item()
X = np.matrix(datadict['feat'])
num_attr = len(datadict['attribute_names'])
num_im = datadict['im_names'].shape[0]
attr_weights = []

for m in xrange(num_attr):
    w = np.load("%s/weights_%d_%s.npy" % (zero_shot_weights_directory, m + 1, datadict['attribute_names'][m]))
    attr_weights.append(w.T.tolist()[0])
    # attr_weights.append(w)

# attr_weights = np.matrix(np.array(attr_weights).reshape((num_attr, w.shape[0])))
attr_weights = np.matrix(attr_weights)
X_attr = np.zeros((num_im, num_attr))

for i in xrange(num_im):
    if datadict['class_labels'][i] not in seen and datadict['class_labels'][i] not in unseen:
        continue
    for m in xrange(num_attr):
        X_attr[i, m] = (attr_weights[m, :] * X[i, :].T)[0, 0]

mean_dict = {}
covariance_dict = {}

means = []
covariances = []

for seen_cat in seen:

    data = X_attr[datadict['class_labels'] == seen_cat, :]
    gmm = GMM().fit(data)
    # print gmm.means_[0]
    mean_dict[seen_cat] = gmm.means_[0]
    covariance_dict[seen_cat] = gmm.covariances_[0]

    means.append(gmm.means_[0].tolist())
    covariances.append(gmm.covariances_[0].tolist())


means = np.array(means)
means = np.sort(means, axis = 0)

dms = np.zeros((num_attr,))
for i in xrange(1, len(seen)):
    dms += means[i, :] - means[i - 1, :]
print means.tolist()
print dms
dms = dms / (len(seen) - 1)
print dms

covariances = np.array(covariances)
mean_covars = np.mean(covariances, axis = 0)

for unseen_cat in unseen:
    new_mean = np.zeros((num_attr,))
    new_covar = mean_covars

    for attr in xrange(num_attr):
        l, r = relative_input[unseen_cat][attr]

        if l == -1 and r == -1:
            pass
        elif l == -1:
            new_mean[attr] = mean_dict[r][attr] - dms[attr]
        elif r == -1:
            new_mean[attr] = mean_dict[l][attr] + dms[attr]
        else:
            new_mean[attr] = (mean_dict[l][attr] + mean_dict[r][attr]) / 2.0

    mean_dict[unseen_cat] = new_mean
    # mean_dict[unseen_cat] = mean_dict[1] + dms
    # covariance_dict[unseen_cat] = new_covar
    covariance_dict[unseen_cat] = covariance_dict[1]

# test_X = np.array( [X[idx].tolist()[0] for idx, l in enumerate(datadict['class_labels']) if l in unseen ] )
test_X_attr = X_attr [ [idx for idx, l in enumerate(datadict['class_labels']) if l in unseen ], : ]

all_cat = seen + unseen
score_predictions = np.zeros((len(all_cat), test_X_attr.shape[0]))

for idx, cat in enumerate(all_cat):
    score_predictions[idx, :] = multivariate_normal.pdf(test_X_attr[10], mean = mean_dict[cat], cov = covariance_dict[cat])

# print score_predictions[-1]
