import numpy as np
import scipy
from rank_svm import *

datadict = np.load('../saved_data/datadict.npy').item()
X = np.matrix(datadict['feat'])
num_attr = len(datadict['attribute_names'])

tot = 0
correct = 0


attr_weights = []
for m in xrange(num_attr):
    w = np.load("../saved_data/weights_ada/weights_%d_%s.npy" % (m + 1, datadict['attribute_names'][m]))
    attr_weights.append(w.T.tolist()[0])

attr_weights = np.matrix(attr_weights)

attr_weights = datadict['relative_att_predictor'].T


for idx, attr in enumerate(datadict['attribute_names']):
    cat_attr_score = datadict['relative_ordering'][idx]
    ranker = attr_weights[idx]

    for i, im1_lab in enumerate(datadict['class_labels']):
        if datadict['used_for_training'][i]:
            continue

        im1_lab -= 1
        for j, im2_lab in enumerate(datadict['class_labels'][i+1:]):
            if datadict['used_for_training'][i]:
                continue
            im2_lab -= 1

            if cat_attr_score[im1_lab] == cat_attr_score[im2_lab]:
                continue

            tot += 1
            im1_score = (ranker * X[i].T)[0, 0]
            im2_score = (ranker * X[j].T)[0, 0]

            if cat_attr_score[im1_lab] < cat_attr_score[im2_lab] and im1_score < im2_score:
                correct += 1
            elif cat_attr_score[im1_lab] > cat_attr_score[im2_lab] and im1_score > im2_score:
                correct += 1


    print correct, tot, 1.0*correct / tot
