# -*- coding: utf-8 -*-
from __future__ import division

""" 
Trains a ResNeXt Model on FEATURE Vectors

"""

__author__ = "Michael Suarez"
__email__ = "masv@connect.ust.hk"
__copyright__ = "Copyright 2019, Hong Kong University of Science and Technology"
__license__ = "3-clause BSD"

import argparse
import os
import json
import pickle
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from models.model011 import FeatureResNeXt
import torch.utils.data as utils
import numpy as np
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains ResNeXt on FEATURE Vectors', formatter_class= argparse.ArgumentDefaultsHelpFormatter)
    # Positional arguments
    parser.add_argument('features_data_path', type=str, help='Root for Features Dict.')
    parser.add_argument('data_folder', type=str, help='Root for Data.')

    # Optimization options
    parser.add_argument('--epochs', '-e', type=int, default=20, help='Number of epochs to train.')
    parser.add_argument('--batch_size', '-b', type=int, default=256, help='Batch size.')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.01, help='The Learning Rate.') #def 0.1
    parser.add_argument('--momentum', '-m', type=float, default=0.3, help='Momentum.')
    parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
    parser.add_argument('--schedule', type=int, nargs='+', default=[10, 15], help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
    # Checkpoints
    parser.add_argument('--save', '-s', type=str, default='./outputmodels', help='Folder to save checkpoints.')
    # Architecture
    parser.add_argument('--depth', type=int, default=65, help='Model depth - Multiple of 3*no_stages (5, 10)')
    parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group) in the DxD convolutionary layer.')
    parser.add_argument('--base_width', type=int, default=8, help='Number of channels in each group. Output of the first convolution layer. Modify stages in model.py')
    parser.add_argument('--widen_factor', type=int, default=4, help='Widen factor between every block')
    # Acceleration
    parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
    parser.add_argument('--prefetch', type=int, default=8, help='Pre-fetching threads.')
    # i/o
    parser.add_argument('--name', type=str, default='model', help='Name your model')
    parser.add_argument('--checkpoint', type=bool, default=False, help='Load from checkpoint?')
    args = parser.parse_args()

    # Init logger
    if not os.path.isdir(args.save):
        os.makedirs(args.save)
    log = open(os.path.join(args.save, args.name + '_log.txt'), 'w')
    state = {k: v for k, v in args._get_kwargs()}
    log.write(json.dumps(state) + '\n')

    # Calculate number of epochs wrt batch size
    args.epochs = args.epochs * args.batch_size // args.batch_size
    args.schedule = [x * args.batch_size // args.batch_size for x in args.schedule]
    
    #initialise Dataset
    # Binary molecules 
    test_b = pickle.load(open("%s/test_b_Vec.mtr" %args.data_folder, "rb"))
    train_b = pickle.load(open("%s/train_b_Vec.mtr" %args.data_folder, "rb"))
    test_nb = pickle.load(open("%s/test_nb_Vec.mtr" %args.data_folder, "rb"))
    train_nb = pickle.load(open("%s/train_nb_Vec.mtr" %args.data_folder, "rb"))
    # Indices of training and testing FF
    test_FF = pickle.load(open("%s/test_FF.mtr" %args.data_folder, "rb"))
    train_FF = pickle.load(open("%s/train_FF.mtr" %args.data_folder, "rb"))
    # Fragfeaturevectors
    Features_all = pickle.load(open(args.features_data_path, "rb"))
    
    # multiplicate Featurevectors
    train_FFAll = Features_all[train_FF]
    test_FFAll = Features_all[test_FF]

    print("loaded data")
    
    #reduce size of dataset for initial testing
#     cutTrain = 2**17
#     np.random.seed(0)
#     selTrain = np.random.choice(train_FF.shape[0], cutTrain, replace=False)
#     cutTest = 2**13
#     np.random.seed(0)
#     selTest = np.random.choice(test_FF.shape[0], cutTest, replace=False)
#     train_FFAll = train_FFAll[selTrain]
#     test_FFAll = test_FFAll[selTest]
#     train_b = train_b[selTrain]
#     train_nb = train_nb[selTrain]
#     test_b = test_b[selTest]
#     test_nb = test_nb[selTest]
    
    #validation split 1%
    np.random.seed(0)
    #ss = np.random.choice(range(train_FFAll.shape[0]), int(0.01*train_FFAll.shape[0]), replace=False)
    #val_FFAll = train_FFAll[ss]
    #val_b = train_b[ss]
    #val_nb = train_nb[ss]    
    #train_FFAll = np.delete(train_FFAll, ss, 0)
    #train_b = np.delete(train_b, ss, 0)
    #train_nb = np.delete(train_nb, ss, 0)

    #normalise Featurevectors
    mean = [train_FFAll[:,i].mean() for i in range(480)]
    std = [train_FFAll[:,i].std() for i in range(480)]
    pickle.dump(mean, open("COVID_mean.mtr", "wb"))
    pickle.dump(std, open("COVID_std.mtr","wb"))
                           
