"""
Merges Relevant KB data into one single ALL.SUM data file for input, output and annotations.
"""

__author__ = "Michael Suarez"
__email__ = "masv@connect.ust.hk"
__copyright__ = "Copyright 2019, Hong Kong University of Science and Technology"
__license__ = "3-clause BSD"


import pickle
import numpy as np
import time
from argparse import ArgumentParser

parser = ArgumentParser(description="Build Files")
parser.add_argument("--datadir", type=str, default="Data", help="input - XXX.YYY ")
parser.add_argument("--envNewAcronym", type=str, default="PRT.SNW", help="input - XXX.YYY ")

args = parser.parse_args()

env = 'ALI.CT' # ALI.CT, ARG.CZ, ARO.PSEU, CON.PSEU, COO.PSEU, HIS.PSEU, HYD.OH, LYS.NZ, PRO.PSEU, RES.N, RES.O, TRP.NE1
env2 = 'ARG.CZ'
env3 = 'ARO.PSEU'
env4 = 'CON.PSEU'
env5 = 'COO.PSEU'
env6 = 'HIS.PSEU'
env7 = 'HYD.OH'
env8 = 'LYS.NZ'
env9 = 'PRO.PSEU'
env10 = 'RES.N'
env11 = 'RES.O'
env12 = 'TRP.NE1'
envN = args.envNewAcronym

string1 = 'Homogenised.annotation.txt'
string2 = 'Homogenised.boundfrags.txt'
string3 = 'Homogenised.property.pvar'

def filenms(filetype):
    filenames = ['%s/%s.%s'  %(env, env, filetype), '%s/%s.%s' %(env2, env2, filetype), '%s/%s.%s' %(env3, env3, filetype), '%s/%s.%s' %(env4, env4, filetype), '%s/%s.%s' %(env5, env5, filetype), '%s/%s.%s' %(env6, env6, filetype), '%s/%s.%s' %(env7, env7, filetype), '%s/%s.%s' %(env8, env8, filetype), '%s/%s.%s' %(env9, env9, filetype), '%s/%s.%s' %(env10, env10, filetype), '%s/%s.%s' %(env11, env11, filetype), '%s/%s.%s' %(env12, env12, filetype)]
    return filenames

trainNew = pickle.load(open("../%s/" %(args.datadir) +filenms(string3)[0], "rb"))
print(trainNew.shape)
for i in filenms(string3)[1:]:
    train2 = pickle.load(open("../%s/" %(args.datadir) +i, "rb"))
    trainNew = np.append(trainNew, train2, axis =0)
    print(trainNew.shape)
pickle.dump(trainNew, open("../%s/" %(args.datadir) +"%s/%s.Homogenised.property.pvar" %(envN, envN), "wb"))
    
with open("../%s/" %(args.datadir) +'%s/%s.Homogenised.annotation.txt'  %(envN, envN), 'w') as outfile:
    for fname in filenms(string1):
        with open("../%s/" %(args.datadir)+fname) as infile:
            outfile.write(infile.read())
            
with open("../%s/" %(args.datadir)+'%s/%s.Homogenised.boundfrags.txt'  %(envN, envN), 'w') as outfile:
    for fname in filenms(string2):
        with open("../%s/" %(args.datadir) +fname) as infile:
            outfile.write(infile.read())        

print('bound created')
