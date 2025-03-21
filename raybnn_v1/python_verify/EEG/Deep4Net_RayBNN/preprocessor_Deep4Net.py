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
import torch
import torch.nn.functional as F
from braindecode.datautil.signal_target import SignalAndTarget
from braindecode.models.deep4 import Deep4Net
from braindecode.torch_ext.optimizers import AdamW
from braindecode.torch_ext.util import set_random_seeds
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

from torchinfo import summary


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
torch.cuda.set_device(args.gpu)
set_random_seeds(seed=20200205, cuda=True)
BATCH_SIZE = 16
TRAIN_EPOCH = 200  # consider 200 for early stopping

# Get data from single subject.


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



    # final_conv_length = auto ensures we only get a single output in the time dimension
    model = Deep4Net(in_chans=in_chans, n_classes=n_classes,
                     input_time_length=train_set.X.shape[2],
                     final_conv_length='auto').cuda()


    checkpoint = torch.load( pjoin(
        outpath, 'model_f{}_cv{}.pt'.format(fold, cv_index)))

    model.network.load_state_dict(checkpoint['model_state_dict'])
    # Only optimize parameters that requires gradient.

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.network.parameters()),
                      lr=1*0.01, weight_decay=0.5*0.001)
    model.compile(loss=F.nll_loss, optimizer=optimizer,
                  iterator_seed=20200205, )

    model.fit(X_train, Y_train, 0, 200)

    #summary(model.network, input_size=(200, 62, 1000, 1))








    activation2 = []
    def get_activation2(name):
        def hook(model, input, output):
            activation2.append(output.cpu().detach().numpy()) 
        return hook
    
    model.network.pool_4.register_forward_hook(get_activation2("aa"))
    model.predict_classes(train_set.X)
    
    outdata = []
    for qq in range(len(activation2)):
        item = activation2[qq]
        for vv in range(item.shape[0]):
            outdata.append(item[vv,:,:,:].flatten())
    outdata = np.array(outdata)
    print(outdata.shape)
    print(Y_train.shape)

    X_max = np.max(outdata)
    X_min = np.min(outdata)
    X_mean = (X_max + X_min)/2

    outdata = (outdata - X_mean)/(X_max - X_min)

    np.savetxt(outpath+"/X_train_"+str(fold)+".txt", outdata, fmt='%1.8e', delimiter=',') 
    np.savetxt(outpath+'/Y_train_'+str(fold)+'.txt', Y_train, fmt='%1.8e', delimiter=',')












    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.cpu().detach().numpy()
        return hook
    
    model.network.pool_4.register_forward_hook(get_activation("conv_4"))

    y_pred = model.predict_classes(test_set.X)


    print("data")

    CNN_output = activation['conv_4']
    outdata = []
    for qq in range(CNN_output.shape[0]):
        outdata.append(CNN_output[qq,:,:,:].flatten())
    outdata = np.array(outdata)
    print(outdata.shape)
    print(Y_test[200:].shape)

    outdata = (outdata - X_mean)/(X_max - X_min)

    np.savetxt(outpath+"/X_test_"+str(fold)+".txt", outdata, fmt='%1.8e', delimiter=',') 
    np.savetxt(outpath+'/Y_test_'+str(fold)+'.txt', Y_test[200:], fmt='%1.8e', delimiter=',')

    p, r, f1, _ = precision_recall_fscore_support(test_set.y, y_pred, average='macro')
    acc = accuracy_score(test_set.y, y_pred)

    roc = roc_auc_score(test_set.y, model.predict_outs(test_set.X)[:, 1])
    print(acc)

    info.append( np.array([TRAIN_EPOCH, acc, p, r, f1, roc])  )














    activation3 = []
    def get_activation3(name):
        def hook(model, input, output):
            activation3.append(output.cpu().detach().numpy()) 
        return hook
    
    model.network.pool_4.register_forward_hook(get_activation3("aa"))
    model.predict_classes(valid_set.X)
    
    outdata = []
    for qq in range(len(activation3)):
        item = activation3[qq]
        for vv in range(item.shape[0]):
            outdata.append(item[vv,:,:,:].flatten())
    outdata = np.array(outdata)
    print(outdata.shape)
    print(Y_val.shape)

    outdata = (outdata - X_mean)/(X_max - X_min)

    np.savetxt(outpath+"/X_val_"+str(fold)+".txt", outdata, fmt='%1.8e', delimiter=',') 
    np.savetxt(outpath+'/Y_val_'+str(fold)+'.txt', Y_val, fmt='%1.8e', delimiter=',')












    shapez = []
    shapez.append([X_train.shape[0], outdata.shape[1] ])
    shapez.append([X_val.shape[0], outdata.shape[1] ])
    shapez.append([Y_test[200:].shape[0], outdata.shape[1] ])

    shapez = np.array(shapez)
    np.savetxt(outpath+'/shape_'+str(fold)+'.txt', shapez, delimiter=',', fmt='%i')


    break


info = np.array(info)
np.savetxt(outpath+"/CNN_info_"+str(fold)+".txt", info, delimiter=',') 


