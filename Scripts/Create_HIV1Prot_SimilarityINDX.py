"""
Creates indices of HIV-1 Protease
"""

__author__ = "Michael Suarez"
__email__ = "masv@connect.ust.hk"
__copyright__ = "Copyright 2019, Hong Kong University of Science and Technology"
__license__ = "3-clause BSD"

from argparse import ArgumentParser
import numpy as np
import pickle

parser = ArgumentParser(description="Build Files")
parser.add_argument("annotation", type=str, help="input - _._.annotation.txt")

args = parser.parse_args()

# collects the environment names in order
coll = []
with open(args.annotation, "r") as f:
    for line in f:
        coll.append(line.split()[0][4:8])

print('Environments in Dataset: %s' %(len(coll)))

# HIV-1 Protease
comp = ['1HTE', '3K4V', '4FLG', '4L1A', '3MWS', '3MXD', '3MXE', '3OQA', '3OQD', '3OQ7', '4EYR', '3H5B', '3HAU', '4PHV', '4NKK', '4JEC', '4K4P', '4K4Q', '4HDB', '4HDF', '4HEG', '4HE9', '4HDP', '4HVP', '3D1X']

print('HIV1 Protease: %s' %(len(comp)))

# collects the environments exact same to the comp array by index
sums2 = []
for j in comp:
    if j.lower() in coll:
        sums2.append([i for i, e in enumerate(coll) if e == j.lower()])
indx_list2 = [item for sublist in sums2 for item in sublist]

print('Same within the Dataset: %s' %(len(indx_list2)))


pickle.dump(indx_list2, open('%sHIV1ProteaseIndx.mtr' %(args.annotation[:-14]), "wb"))

