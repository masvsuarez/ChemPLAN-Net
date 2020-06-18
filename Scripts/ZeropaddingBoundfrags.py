"""
Use the boundfrags.txt file and applies zero-padding to create a square matrix -> for the KB
"""

__author__ = "Michael Suarez"
__email__ = "masv@connect.ust.hk"
__copyright__ = "Copyright 2019, Hong Kong University of Science and Technology"
__license__ = "3-clause BSD"

import numpy as np

from argparse import ArgumentParser

parser = ArgumentParser(description="Build Files")
parser.add_argument("--datadir", type=str, default="Data", help="input - XXX.YYY ")
parser.add_argument("--envNewAcronym", type=str, default="PRT.SNW", help="input - XXX.YYY ")

args = parser.parse_args()

with open("../%s/%s/%s.Homogenised.boundfrags.txt" %(args.datadir, args.envNewAcronym, args.envNewAcronym), "r") as f:
    linecount = 0
    maxcount = 0
    for line in f:
        if len(line.split()) > maxcount:
            maxcount = len(line.split())
        linecount+=1

matr = np.zeros((linecount, maxcount), dtype=int)
with open("../%s/%s/%s.Homogenised.boundfrags.txt" %(args.datadir, args.envNewAcronym, args.envNewAcronym), "r") as f:
    for j, line in enumerate(f):
        temp = line.split()
        matr[j, 0:len(temp)] = list(map(int, temp))
np.savetxt("../%s/%s/%s.Homogenised.boundfrags_zeros.txt" %(args.datadir, args.envNewAcronym, args.envNewAcronym), matr, delimiter=',', fmt ='%i')
print(matr.shape)
