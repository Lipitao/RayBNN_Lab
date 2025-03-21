#!/usr/bin/env python
# coding: utf-8
'''Subject-independent classification with KU Data,
using Deep ConvNet model from [1].

References
----------
.. [1] Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J.,
   Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F. & Ball, T. (2017).
   Deep learning with convolutional neural networks for EEG decoding and
   visualization.
   Human Brain Mapping , Aug. 2017. Online: http://dx.doi.org/10.1002/hbm.23730
'''

import argparse
import json
import logging
import sys
from os import makedirs
from os.path import join as pjoin
from shutil import copy2, move

import h5py
import numpy as np

from braindecode.datautil.signal_target import SignalAndTarget


from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression


from pyriemann.tangentspace import TangentSpace

from pyriemann.estimation import XdawnCovariances
from pyriemann.classification import MDM

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

from sklearn.metrics import roc_auc_score
logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                    level=logging.INFO, stream=sys.stdout)

parser = argparse.ArgumentParser(
    description='Subject-independent classification with KU Data')
parser.add_argument('datapath', type=str, help='Path to the h5 data file')
parser.add_argument('outpath', type=str, help='Path to the result folder')
parser.add_argument('-fold', type=int,
                    help='k-fold index, starts with 0', required=True)
parser.add_argument('-gpu', type=int, help='The gpu device to use', default=0)

args = parser.parse_args()
datapath = args.datapath
outpath = args.outpath
fold = args.fold
assert(fold >= 0 and fold < 54)
# Randomly shuffled subject.
subjs = [35, 47, 46, 37, 13, 27, 12, 32, 53, 54, 4, 40, 19, 41, 18, 42, 34, 7,
         49, 9, 5, 48, 29, 15, 21, 17, 31, 45, 1, 38, 51, 8, 11, 16, 28, 44, 24,
         52, 3, 26, 39, 50, 6, 23, 2, 14, 25, 20, 10, 33, 22, 43, 36, 30]
test_subj = subjs[fold]
cv_set = np.array(subjs[fold+1:] + subjs[:fold])
kf = KFold(n_splits=6)

dfile = h5py.File(datapath, 'r')

BATCH_SIZE = 16
TRAIN_EPOCH = 200  # consider 200 for early stopping

# Get data from single subject.
print(len(subjs))

def get_data(subj):
    dpath = '/s' + str(subj)
    X = dfile[pjoin(dpath, 'X')]
    Y = dfile[pjoin(dpath, 'Y')]
    return X, Y


def get_multi_data(subjs):
    Xs = []
    Ys = []
    for s in subjs:
        x, y = get_data(s)
        Xs.append(x[:])
        Ys.append(y[:])
    X = np.concatenate(Xs, axis=0)
    Y = np.concatenate(Ys, axis=0)
    return X, Y



info = []
for cv_index, (train_index, test_index) in enumerate(kf.split(cv_set)):

    train_subjs = cv_set[train_index]
    valid_subjs = cv_set[test_index]
    X_train, Y_train = get_multi_data(train_subjs)
    X_val, Y_val = get_multi_data(valid_subjs)
    X_test, Y_test = get_data(test_subj)
    train_set = SignalAndTarget(X_train, y=Y_train)
    valid_set = SignalAndTarget(X_val, y=Y_val)
    test_set = SignalAndTarget(X_test[200:], y=Y_test[200:])
    n_classes = 2
    in_chans = train_set.X.shape[1]

    print(X_train.shape)
    print(Y_train.shape)

    for n_components in range(1,15):
        #clf = make_pipeline(XdawnCovariances(n_components), MDM())

        clf = make_pipeline(XdawnCovariances(n_components), TangentSpace(metric='riemann'), LogisticRegression(max_iter=1000))


        clf.fit(X_train, Y_train)
        y_pred = clf.predict(X_test[200:])

        p, r, f1, _ = precision_recall_fscore_support(Y_test[200:], y_pred, average='macro')
        acc = accuracy_score(Y_test[200:], y_pred)

        roc = roc_auc_score(Y_test[200:], clf.predict_proba(X_test[200:])[:, 1])

        print(n_components)
        print(acc)

        info.append( np.array([n_components, acc, p, r, f1, roc])  )
    break

info = np.array(info)
np.savetxt(outpath+"/LR_info_"+str(fold)+".txt", info, delimiter=',') 





