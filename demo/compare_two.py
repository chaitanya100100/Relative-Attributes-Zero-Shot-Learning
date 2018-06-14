import numpy as np
import sys

if len(sys.argv) != 4:
    print "Usage : %s <img1_name> <img2_name> <attribute_name>" % sys.argv[0]
    exit(-1)

im1_name = sys.argv[1]
im2_name = sys.argv[2]
attr_name = sys.argv[3]

datadict = np.load('../datadict.npy').item()
im_names = datadict['im_names'].tolist()
attribute_names = datadict['attribute_names']

if im1_name not in im_names:
    print "%s image is not there" % im1_name
    exit(-1)

if im2_name not in im_names:
    print "%s image is not there" % im2_name
    exit(-1)

if attr_name not in attribute_names:
    print "%s attribute is not there" % attr_name
    print "attributes are :"
    print attribute_names
    exit(-1)

attr_idx = attribute_names.index(attr_name)
im1_idx = im_names.index(im1_name)
im2_idx = im_names.index(im2_name)


print "attribute index :", attr_idx
print "img1 index :", im1_idx
print "img2 index :", im2_idx

ranker_given = datadict['relative_att_predictor'][:, attr_idx]
ranker = np.load("../weights_ada/weights_%d_%s.npy" % (attr_idx + 1, attr_name))

X = datadict['feat']
im1_score =  np.dot(X[im1_idx], ranker)
im2_score = np.dot(X[im2_idx], ranker)

print "(ours )im1 score : %f\tim2 score : %f" % (im1_score, im2_score)
print "(given)im1 score : %f\tim2 score : %f" % (np.dot(X[im1_idx], ranker_given), np.dot(X[im2_idx], ranker_given))

if im1_score > im2_score:
    print "%s shows more %s attribute than %s" % (im1_name, attr_name, im2_name)
else:
    print "%s shows more %s attribute than %s" % (im2_name, attr_name, im1_name)
