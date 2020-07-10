# -*- coding: utf-8 -*-
from __future__ import division

""" 
Generates the Normalisation Values for the training set FEATURE Vectors

"""
__author__ = "Michael Suarez"
__email__ = "masv@connect.ust.hk"
__copyright__ = "Copyright 2019, Hong Kong University of Science and Technology"
__license__ = "3-clause BSD"

import argparse
import os
import json
import pickle
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generates Mean/STD', formatter_class= argparse.ArgumentDefaultsHelpFormatter)
    # Positional arguments
    parser.add_argument('features_data_path', type=str, help='Root for Features Dict.')
    parser.add_argument('data_folder', type=str, help='Root for Data.')
    args = parser.parse_args()

    #initialise Dataset
    # Binary molecules 
    train_b = pickle.load(open("%s/train_b_Vec.mtr" %args.data_folder, "rb"))
    train_nb = pickle.load(open("%s/train_nb_Vec.mtr" %args.data_folder, "rb"))
    # Indices of training FF
    train_FF = pickle.load(open("%s/train_FF.mtr" %args.data_folder, "rb"))
    # Fragfeaturevectors
    Features_all = pickle.load(open(args.features_data_path, "rb"))
    
    # multiplicate Featurevectors
    train_FFAll = Features_all[train_FF]

    print("loaded data")

    #normalise Featurevectors
    FFmean = [train_FFAll[:,i].mean() for i in range(480)]
    FFstd = [train_FFAll[:,i].std() for i in range(480)]
    pickle.dump(FFmean, open("%s/COVID_mean.mtr" %args.data_folder, "wb"))
    pickle.dump(FFstd, open("%s/COVID_std.mtr" %args.data_folder, "wb"))
                           
