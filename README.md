# Relative-Attributes
Python Implementation of Visual Relative Attributes for Image Classification and Zero Shot Learning

## Description
This implementation refers to paper `Relative Attributes, D. Parikh and K. Grauman, International Conference on Computer Vision (ICCV), 2011`. Original code given by authors was in matlab. This repo contains python code for learning relative ranking function using Newton optimization implemented from scratch. Also Zero Shot Learning with Gaussian Mixture Model is implemented in python.

## Implementation Details
- ![src/rank_svm.py](src/rank_svm.py) contains the implementation of rank svm using Newton's method.
- ![src/zero_shot_training.py](src/zero_shot_training.py) and ![src/zero_shot_GMM.py](src/zero_shot_GMM.py) are training and testing files for zero shot learning respectively.
- Pre-extracted gist features from 'PubFig' dataset are used in this implementation. To train on new dataset, ![gist](gist) module and ![src/extract_feature.m](src/extract_feature.m) can be used to extract gist features.
- Learned ranking function, preprocessed data, etc. are read and saved in ![demo](demo) directory.
