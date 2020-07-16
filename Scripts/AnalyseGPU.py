import os
import json
import pickle
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from Scripts.FNN.models.model011 import FeatureResNeXt
import torch.utils.data as utils
import numpy as np
import time
from argparse import ArgumentParser
import sys

'''
Checks the query environments for matching fragments and returns a list of probabilities 
'''

parser = ArgumentParser(description="Build Files")
parser.add_argument('interval', type=int, nargs='+', help='Decrease learning rate at these epochs.')
parser.add_argument("--datadir", type=str, default="Data", help="Data directory")
parser.add_argument('--save', '-s', type=str, default='./outputmodels', help='Folder with trained model')
parser.add_argument('--name', type=str, default='model', help='Your models name')
args = parser.parse_args()


# Option 1: testing the existing FF Vectors from the co-crystal structures
#load the query feature vectors 
#Features_all = pickle.load(open("DataP_New/PRT.SNW/PRT.SNW.Homogenised.property.pvar", "rb"))
#query = pickle.load(open("DataP_New/PRT.SNW/PRT.SNW.Homogenised.HIV1ProteaseIndx.mtr", "rb")) #228

#testing = Features_all[query]

# Option 2: testing HIV_Protease f_pocket output

AQ = pickle.load(open("%s/AllQuery.df" %args.datadir, "rb"))
testing = AQ.iloc[:,8:].values.astype(float)

testing = testing[args.interval[0]:args.interval[1]]

# normalise query input FF Vectors
start = time.time()
mean = pickle.load(open("%s/COVID_mean.mtr" %args.datadir, "rb"))
std = pickle.load(open("%s/COVID_std.mtr" %args.datadir, "rb"))
for i in range(480):
    if std[i] != 0:
        testing[:,i] = (testing[:,i] - mean[i])/std[i]
    else:
        testing[:,i] = testing[:,i]
end = time.time()
print("FF normalising time: ", end-start)

testing_FFAll = np.resize(testing, (testing.shape[0], 1, 6, 80))

# convert to tensor
test_tensor_x = torch.from_numpy(testing_FFAll)

# Converting the Fragment Database into BitVectors
normalDF = pickle.load(open("../EnsembleModel/Data/GrandCID.dict", "rb"))
from rdkit.Chem import AllChem
def morgfing(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
normalDF['fp'] = normalDF['Mol'].apply(morgfing)
moleculeDB = normalDF.iloc[:]['fp'].values

test_tensor_mDB = torch.stack([torch.Tensor(i) for i in moleculeDB])

# duplicate each FF Vector 59732 times (for each fragment once) [0,0,0,0 ...,0, 1, 1, 1, 1, 1, ...]
test_tensor_x = test_tensor_x.repeat(1, 59732, 1, 1).view(-1, 1, 6,80)

print(test_tensor_x.shape)

# duplicate the Fingerprints in blocks [0,1,2,3,4,....59731,0,1,2,3,4,....]
test_tensor_mDB = test_tensor_mDB.repeat(testing.shape[0],1).view(-1, 1024)

print(test_tensor_mDB.shape)

#load the data in the relevant formats
test_data = utils.TensorDataset(test_tensor_x, test_tensor_mDB)

test_loader = utils.DataLoader(test_data, batch_size=512, shuffle=False, num_workers=8, pin_memory=True)

print("Loaded Data")

# load the network with correct parameters
net = FeatureResNeXt(8, 65, 8, 4)
loaded_state_dict = torch.load(os.path.join(args.save, args.name + '_final.pytorch'))
#update model with weights from trained model
temp = {}
for key, val in list(loaded_state_dict.items()):
    # parsing keys for ignoring 'module.' in keys
    temp[key[7:]] = val
loaded_state_dict = temp
net.load_state_dict(loaded_state_dict)
net = net.eval()

net = torch.nn.DataParallel(net, device_ids=list(range(8)))
net.cuda()


# change according to number of environments here and colllect output
coll = np.empty(59732*testing.shape[0])
for batch_idx, (X,X2) in enumerate(test_loader):
    print(batch_idx)
    X, X2 = torch.autograd.Variable(X.cuda()), torch.autograd.Variable(X2.cuda())
    output = net(X.float(), X2.float())
    try:
        coll[batch_idx*512:batch_idx*512+512] = output.data.cpu().numpy().reshape(X.shape[0])
    except:
        coll[batch_idx*512:] = output.data.cpu().numpy().reshape(X.shape[0])
        
print("Done. Now saving")
pickle.dump(coll, open("%s/%s_Probabilities_noEnv%s_%s.mat"%(args.datadir, args.name, args.interval[0], args.interval[1]), "wb"))
