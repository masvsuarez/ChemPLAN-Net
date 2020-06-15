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


Envmt = 'DataP_New/PRT.SNW/PRT.SNW.Homogenised.boundfrags_zeros.txt'
#../EnsembleModel/Data/KIN.ALL/KIN.ALL.Homogenised.boundfrags_zeros.txt

#BoundFrags = np.loadtxt(Envmt, delimiter=',')
binding = pickle.load(open("DataP_New/PRT.SNW/PRT.SNW.binding.mtr", "rb"))

normalDF = pickle.load(open("../EnsembleModel/Data/GrandCID.dict", "rb"))

# binding = np.full(BoundFrags.shape,-1)
# for r, i in enumerate(BoundFrags):
#     for c, j in enumerate(i[i!=0]):
#         binding[r,c]=normalDF.index.get_loc(j)
        
notbinding = np.full(binding.shape, -1)
for j,i in enumerate(binding):
    temp = i[i!=-1]
    bind = list(normalDF.iloc[temp]['Mol'])
    fpsbind = [Chem.RDKFingerprint(x) for x in bind]
    for k,l in enumerate(fpsbind):
        intlist = np.random.randint(59732,size=30)
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

pickle.dump(notbinding, open("DataP_New/PRT.SNW/PRT.SNW.notbinding.mtr", "wb"))
