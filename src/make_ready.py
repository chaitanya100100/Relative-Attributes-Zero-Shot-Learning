import numpy as np
import scipy
from scipy.sparse import csr_matrix
from rank_svm import *
from scipy.io import savemat

datadict = np.load('../saved_data/datadict.npy').item()
X = datadict['feat']

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
        if not datadict['used_for_training'][i]:
            continue

        im1_lab -= 1
        for j, im2_lab in enumerate(datadict['class_labels'][i+1:]):
            if not datadict['used_for_training'][i]:
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
    C_O = scipy.matrix(0.1 * np.ones([O_cnt, 1]))
    C_S = scipy.matrix(0.1 * np.ones([S_cnt, 1]))
    X = scipy.matrix(X)
    savemat("X.mat",{'X':X});
    savemat("S.mat",{'S':S});
    savemat("O.mat",{'O':O});
    savemat("C_O.mat",{'C_O':C_O});
    savemat("C_S.mat",{'C_S':C_S});
    #w = rank_svm(X, S, O, C_S, C_O)
    #np.save("../saved_data/weights_local/weights_%d_%s" % (idx + 1, datadict['attribute_names'][idx]), w)
    #print w
    #print w.shape
    # now train ranksvm for only one attribute, we can extend it later for all attribute
    # break
