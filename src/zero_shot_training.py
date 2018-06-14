import numpy as np
import scipy
from scipy.sparse import csr_matrix
from rank_svm import *
from zero_shot_config import *
import os

datadict = np.load('../saved_data/datadict.npy').item()
X = datadict['feat']

if not os.path.isdir(zero_shot_weights_directory):
    os.mkdir(zero_shot_weights_directory)

for idx, attr in enumerate(datadict['attribute_names']):
    cat_ordering = datadict['relative_ordering'][idx]
    sorted_cat_idx = np.argsort(cat_ordering)

    """
    for i, lesser in enumerate(sorted_cat_idx):
        for greater in sorted_cat_idx[i:]:
            print lesser, greater
    """
    S_row = []
    S_column = []
    S_value = []
    S_cnt = 0
    O_row = []
    O_column = []
    O_value = []
    O_cnt = 0
    for i, im1_lab in enumerate(datadict['class_labels']):
        if im1_lab not in seen:
            continue

        im1_lab -= 1
        for j, im2_lab in enumerate(datadict['class_labels'][i+1:]):
            if im2_lab not in seen:
                continue

            im2_lab -= 1
            # rnum = np.random.rand()
            # if rnum > 0.2:
            #     continue
            if cat_ordering[im1_lab] == cat_ordering[im2_lab]:
                # print i, im1_lab, j, im2_lab
                S_row.append(S_cnt)
                S_column.append(i)
                S_value.append(-1)
                S_row.append(S_cnt)
                S_column.append(i + j + 1)
                S_value.append(1)
                S_cnt += 1

                S_row.append(S_cnt)
                S_column.append(i)
                S_value.append(1)
                S_row.append(S_cnt)
                S_column.append(i + j + 1)
                S_value.append(-1)
                S_cnt += 1

            elif cat_ordering[im1_lab] < cat_ordering[im2_lab]:
                O_row.append(O_cnt)
                O_column.append(i)
                O_value.append(-1)
                O_row.append(O_cnt)
                O_column.append(i + j + 1)
                O_value.append(1)
                O_cnt += 1
            elif cat_ordering[im1_lab] > cat_ordering[im2_lab]:
                O_row.append(O_cnt)
                O_column.append(i)
                O_value.append(1)
                O_row.append(O_cnt)
                O_column.append(i + j + 1)
                O_value.append(-1)
                O_cnt += 1


    S = csr_matrix((S_value, (S_row, S_column)),(S_cnt, datadict['feat'].shape[0]))
    O = csr_matrix((O_value, (O_row, O_column)),(O_cnt, datadict['feat'].shape[0]))
    print S.shape
    print O.shape
    C_O = scipy.matrix(0.1 * np.ones([O_cnt, 1]))
    C_S = scipy.matrix(0.1 * np.ones([S_cnt, 1]))
    X = scipy.matrix(X)
    w = rank_svm(X, S, O, C_S, C_O)
    np.save("%s/weights_%d_%s" % (zero_shot_weights_directory, idx + 1, datadict['attribute_names'][idx]), w)
