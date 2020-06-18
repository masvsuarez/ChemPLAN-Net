"""
Creates non-binding data
"""

__author__ = "Michael Suarez"
__email__ = "masv@connect.ust.hk"
__copyright__ = "Copyright 2019, Hong Kong University of Science and Technology"
__license__ = "3-clause BSD"

import pickle
import numpy as np
import pandas as pd
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit import Chem
from argparse import ArgumentParser

parser = ArgumentParser(description="Build Files")
parser.add_argument("--datadir", type=str, default="Data", help="input - XXX.YYY ")
parser.add_argument("--envNewAcronym", type=str, default="PRT.SNW", help="input - XXX.YYY ")

args = parser.parse_args()

#Load Binding Fragments
binding = pickle.load(open("../%s/%s/%s.binding.mtr"  %(args.datadir, args.envNewAcronym, args.envNewAcronym), "rb"))
normalDF = pickle.load(open("../%s/GrandCID.dict" %(args.datadir), "rb"))

#Create Non Binding based on 
notbinding = np.full(binding.shape, -1) # Indices will correspond to the positions in normalDF
for j,i in enumerate(binding):
    temp = i[i!=-1]
    bind = list(normalDF.iloc[temp]['Mol'])
    #Create Fingerprints for all the Fragments
    fpsbind = [Chem.RDKFingerprint(x) for x in bind]
    for k,l in enumerate(fpsbind):
        intlist = np.random.randint(59732,size=50)
        c = list(normalDF.iloc[intlist]['Mol'])
        d = [Chem.RDKFingerprint(x) for x in c]
        for m, n in enumerate(d):
            if DataStructs.FingerprintSimilarity(l,n) < 0.05:
                notbinding[j,k] = int(intlist[m])
                break
            if m == len(d)-1:
                notbinding[j,k] = int(intlist[m])
    if j%1000==0:
        print(j)

pickle.dump(notbinding, open("../%s/%s/%s.nonbinding.mtr"  %(args.datadir, args.envNewAcronym, args.envNewAcronym), "wb"))
