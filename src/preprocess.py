import numpy as np
import scipy.io as sio

# load data.mat file
datadict = sio.loadmat('../saved_data/pubfig_data.mat')

# some variables are in bad format
# process them to make them in correct format

attr_names = [x[0] for x in datadict['attribute_names'][0]]
datadict['attribute_names'] = attr_names

datadict['im_names'] = datadict['im_names'][0]
datadict['class_labels'] = datadict['class_labels'][:, 0]
datadict['used_for_training'] = datadict['used_for_training'][:, 0]
datadict['class_names'] = datadict['class_names'][0]

np.save("../saved_data/datadict.npy", datadict)

"""
For loading back :
d = np.load('../saved_data/datadict.npy').item()
"""
