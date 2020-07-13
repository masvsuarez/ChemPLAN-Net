
"""
Fragmentize the ligands. Produce CID list and smiles.
"""

__author__ = "Jordy Homing Lam"
__copyright__ = "Copyright 2018, Hong Kong University of Science and Technology"
__license__ = "3-clause BSD"


from argparse import ArgumentParser
import pickle
import sys
import os
import multiprocessing
from functools import partial
import subprocess
import glob
import time

from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit import rdBase
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import Crippen
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import SDWriter

#from Supporting import *
#from CythonDissimilarity import dissimilarity_to_KBCython

parser = ArgumentParser(description="Decompose ligands into fragments")
parser.add_argument("--ligandfolder", type=str, dest="ligandfolder",
                    help="folder of input/output files")
parser.add_argument("--datadir", type=str, dest="datadir",
                    help="folder of input/output files")

args = parser.parse_args()


# ===================
# Define Fingerprint
# ===================

FingerprintingFx = lambda x:Chem.PatternFingerprint(x)

# =========================================
# Read in the CID Smiles Dicts
# =========================================

CidSmilesDf = pickle.load(open("%s/GrandCID.dict" %(args.datadir), "rb"))

# =========================================
# Comparison step
# =========================================

#for fn in [i for i in glob.glob("%s/*.sdf" %(args.ligandfolder))]:
def FindMatchCID(fn, CidSmilesDf):
    if os.path.exists("%s/%s/%s.list" %(args.ligandfolder, fn.split('/')[-1].split(".")[0], fn.split('/')[-1].split(".")[0])):
        return print("Already Computed")
    if not os.path.isdir("%s/%s" %(args.ligandfolder, fn.split('/')[-1].split(".")[0])):
        os.makedirs("%s/%s" %(args.ligandfolder, fn.split('/')[-1].split(".")[0]))
    print(fn)
    start = time.time()
    # PdbLig is the pdb ligand cast as mol object
    PdbLig = Chem.MolFromMolFile(fn)
    Chem.AssignAtomChiralTagsFromStructure(PdbLig)
    Chem.AssignStereochemistry(PdbLig, cleanIt=False, force=False, flagPossibleStereoCenters=True)

    # Create Fingerprint For ligand
    fp1 = FingerprintingFx(Chem.RemoveHs(PdbLig))

    match_cid = []
    for index,row in CidSmilesDf.iterrows(): # Takes 20 seconds
        mc = Chem.RemoveHs(row['Mol'])
        fp2 = FingerprintingFx(mc)
        # preliminary check
        if (fp1 & fp2) == fp2:
            if PdbLig.HasSubstructMatch(row['Mol']):
                match_cid.append(index)

    pickle.dump(match_cid, open("%s/%s/%s.list" %(args.ligandfolder, fn.split('/')[-1].split(".")[0], fn.split('/')[-1].split(".")[0]),"wb"))

    print("Finished analysing %s in %s s" %(fn, time.time() - start))

    return fn, match_cid


FindMatchCID_Wrapper = partial(FindMatchCID, CidSmilesDf = CidSmilesDf)
pool = multiprocessing.Pool(processes = 24)
MatchResults = pool.map(FindMatchCID_Wrapper, [i for i in glob.glob("%s/*.sdf" %(args.ligandfolder))])
pool.close()
