import numpy as np
import sys
import os
import scipy.io as sio

if len(sys.argv) != 3:
    print "Usage : %s <img_feature_mat_path> <attribute_name>" % sys.argv[0]
    exit(-1)

feat_path = sys.argv[1]
attr_name = sys.argv[2]

if not os.path.isfile(feat_path):
    print "No file %s exists" % feat_path
    exit(-1)

datadict = np.load('../datadict.npy').item()
attribute_names = datadict['attribute_names']

if attr_name not in attribute_names:
    print "%s attribute is not there" % attr_name
    print "attributes are :"
    print attribute_names
    exit(-1)

attr_idx = attribute_names.index(attr_name)

matdict = sio.loadmat(feat_path)
x = matdict['tot_feat']

ranker_given = datadict['relative_att_predictor'][:, attr_idx]
ranker = np.load("../weights_ada/weights_%d_%s.npy" % (attr_idx + 1, attr_name))

im_score =  np.dot(x, ranker)[0][0]
print im_score
