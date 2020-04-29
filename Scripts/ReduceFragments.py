"""
Remove Fragments not in Knowledgebase
"""

__author__ = "Michael Suarez"
__email__ = "masv@connect.ust.hk"
__copyright__ = "Copyright 2019, Hong Kong University of Science and Technology"
__license__ = "3-clause BSD"

from argparse import ArgumentParser
import numpy as np
import pickle

# Check the Bound Fragments

BoundFrags = np.loadtxt("../DataP_New/PRT.SNW/PRT.SNW.Homogenised.boundfrags_zeros.txt", delimiter=',')

normalDF = pickle.load(open("../EnsembleModel/Data/GrandCID.dict", "rb"))

binding = np.full(BoundFrags.shape,-1)
mlength = 0
for r, i in enumerate(BoundFrags):
    for c, j in enumerate(i[i!=0]):
        try:
            # Checks whether the Fragment can be found in the 59k Fragment Base
            binding[r,c]=normalDF.index.get_loc(int(j))
        except:
            continue
    temp = binding[r]
    if temp[temp!=-1].shape[0] > mlength:
        mlength = temp[temp!=-1].shape[0]
        print(mlength) #Finds the maximum number of Fragments per environment -> 705
        
indices = np.empty(binding.shape[0])
red_binding = np.full((binding.shape[0], mlength), -1)
for j, i in enumerate(binding):
    indices[j] = i[i!=-1].shape[0]
    red_binding[j][:int(indices[j])] = i[i!=-1]
red_binding = np.delete(red_binding, np.where(indices==0), axis=0)

pickle.dump(red_binding, open("../DataP_New/PRT.SNW/PRT.SNW.binding.mtr", "wb"))

# Removes environments without binding Fragments
Features_all = pickle.load(open("DataP_New/PRT.SNW/PRT.SNW.Homogenised.property.pvar", "rb"))
Features_all = np.delete(Features_all, np.where(indices==0), axis=0)
pickle.dump(Features_all, open("DataP_New/PRT.SNW/PRT.SNW.Homogenised.property.pvar_red", "wb"))

# Removes environment annotiation without binding fragments
with open("DataP_New/PRT.SNW/PRT.SNW.Homogenised.annotation.txt", "r+") as f:
    lines = f.readlines()
    for i in np.where(indices==0)[0][::-1]:
        del lines[i]
    f.seek(0)
    f.truncate()
    f.writelines(lines)
