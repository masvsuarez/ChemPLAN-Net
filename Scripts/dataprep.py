import pickle
import numpy as np
import pandas as pd
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit import Chem

Features_all = pickle.load(open("DataP_New/PRT.SNW/PRT.SNW.Homogenised.property.pvar", "rb"))

#Fragments bound per environment

binding = pickle.load(open("DataP_New/PRT.SNW/PRT.SNW.binding.mtr", "rb"))
notbinding = pickle.load(open("DataP_New/PRT.SNW/PRT.SNW.notbinding.mtr", "rb"))


# Indices of relevant HIV-1 Homologs and Identities

HIVIndx = pickle.load(open("DataP_New/PRT.SNW/PRT.SNW.Homogenised.HIV1ProteaseIndx_99.mtr", "rb"))
HIVHomoIndx = pickle.load(open("DataP_New/PRT.SNW/PRT.SNW.Homogenised.HIV1ProteaseIndx_70.mtr", "rb"))


normalDF = pickle.load(open("../EnsembleModel/Data/GrandCID.dict", "rb"))

def morgfing(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
normalDF['fp'] = normalDF['Mol'].apply(morgfing)
normalDF.head()

test_b = binding[HIVIndx]
train_b = np.delete(binding, HIVHomoIndx, 0)
test_nb = notbinding[HIVIndx]
train_nb = np.delete(notbinding, HIVHomoIndx, 0)
print(test_nb.shape,train_nb.shape)

#contains Fragment Indexes and padded with -1
train_b_f = np.empty(train_b.shape[0])
train_nb_f = np.empty(train_nb.shape[0])

test_b_f = np.empty(test_b.shape[0])
test_nb_f = np.empty(test_nb.shape[0])

def counter(datamat, sums):
    for j, i in enumerate(datamat):
        sums[j] = i[i!=-1].shape[0]
    return sums

#contains number of Fragments per environment
train_b_f = counter(train_b, train_b_f)
train_nb_f = counter(train_nb, train_nb_f)

test_b_f = counter(test_b, test_b_f)
test_nb_f = counter(test_nb, test_nb_f)

train_FF = np.empty(int(train_b_f.sum()), dtype=int)
test_FF = np.empty(int(test_b_f.sum()), dtype=int)

def FFcreator(mat_FF, numb, indxmat):
    cum = 0
    for j, i in enumerate(numb):
        i = int(i)
        if j < numb.shape[0]:
            mat_FF[cum:cum+i] = [int(indxmat[j])]*i
            cum += i
        else:
            mat_FF[cum:] = [int(indxmat[j])]*i
    return mat_FF

test_FF = FFcreator(test_FF, test_b_f, HIVIndx)

ProtIndx = range(Features_all.shape[0])
ProtIndx = np.delete(ProtIndx, HIVHomoIndx, axis = 0)
train_FF = FFcreator(train_FF, train_b_f, ProtIndx)

print(test_FF.shape, train_FF.shape)

pickle.dump(train_FF, open("DataP_New/train_FF.mtr", "wb"))
pickle.dump(test_FF, open("DataP_New/test_FF.mtr", "wb"))

#creates the actual Fingerprint Vectors

def createData(FragIndx, numb):
    out = np.empty(int(numb.sum()),dtype=object)
    count = 0
    for j, i in enumerate(FragIndx):
        temp = i[i!=-1]
        if j == len(FragIndx)-1:
            out[count:] = normalDF.iloc[temp]['fp'].values
        else:
            out[count:int(count+numb[j])] =  normalDF.iloc[temp]['fp'].values
        count+=int(numb[j])
    return out

train_b_Vec = createData(train_b, train_b_f)
pickle.dump(train_b_Vec, open("DataP_New/train_b_Vec.mtr", "wb"))

train_nb_Vec = createData(train_nb, train_nb_f)
pickle.dump(train_nb_Vec, open("DataP_New/train_nb_Vec.mtr", "wb"))


test_b_Vec = createData(test_b, test_b_f)
pickle.dump(test_b_Vec, open("DataP_New/test_b_Vec.mtr", "wb"))

test_nb_Vec = createData(test_nb, test_nb_f)
pickle.dump(test_nb_Vec, open("DataP_New/test_nb_Vec.mtr", "wb"))
