##############################################################################
#
# FragFeatureNet
#                                                                     
##############################################################################

__author__ = "Michael Suarez"
__email__ = "masv@connect.ust.hk"
__copyright__ = "Copyright 2019, Hong Kong University of Science and Technology"
__license__ = "3-clause BSD"

#Change things here

PROTEINFAMILYACRONYM = 'PRT.SNW'    # Choose a Acronym in the format of XXX.YYY for your Data
SCRIPTHOME='Scripts'		    # This is the default location of scripts
DATAHOME='Data'				    # This is the pre-processed Data home
MODELHOME='FNN'				    # This is the default location of models
ModelOut="Results_Protease"     #Modeloutput
MODELNUM="BinaryModel_v00_FULL" #Modelname
INQUIRY='Results_test_CDK2'	    # This is the inquiry folder - Currently Not in Use
LigandSourceFolder="Ligands"

#Dont Change
FEATURESPROT='${DATAHOME}/${PROTEINFAMILYACRONYM}/${PROTEINFAMILYACRONYM}.Homogenised.property.pvar'    # This is the default location of the FeatureVectors


###

### Step 1: FuseData.py
# Environments to Fuse
#{"ALI.CT", "ARG.CZ", "ARO.PSEU", "CON.PSEU", "COO.PSEU", "HIS.PSEU", "HYD.OH", "LYS.NZ", "PRO.PSEU", "RES.N", "RES.O", "TRP.NE1"}

python ${SCRIPTHOME}/FuseData.py --datadir ${DATAHOME} --envNewAcronym ${PROTEINFAMILYACRONYM}

### Step 2: ZeropaddingBoundfrags.py

python ${SCRIPTHOME}/ZeropaddingBoundfrags.py --datadir ${DATAHOME} --envNewAcronym ${PROTEINFAMILYACRONYM}

### Step 3: ReduceFragments.py

python ${SCRIPTHOME}/ReduceFragments.py --datadir ${DATAHOME} --envNewAcronym ${PROTEINFAMILYACRONYM}

### Step 4: CreateSimilarityIndex.py

python ${SCRIPTHOME}/CreateSimilarityIndex.py --datadir ${DATAHOME} --envNewAcronym ${PROTEINFAMILYACRONYM}

### Step 5: CreateNonBinding.py

python ${SCRIPTHOME}/CreateNonBinding.py --datadir ${DATAHOME} --envNewAcronym ${PROTEINFAMILYACRONYM}

### Step 6: DataPrep.py

python ${SCRIPTHOME}/DataPrep.py --datadir ${DATAHOME} --envNewAcronym ${PROTEINFAMILYACRONYM}

#===================
# Train-Test BinaryModel for Protease
#===================

srun --nodelist=node-2 nohup python ${SCRIPTHOME}/${MODELHOME}/train_test_model.py ${FEATURESPROT} ${DATAHOME} --save ${ModelOut} --depth 65  -b 512 -lr 0.3 -m 0.01 -d 0.0001 -e 20 --name ${MODELNUM} --ngpu 8 > ${ModelOut}/${MODELNUM}.out &

#===================
# Analyse the Model
#===================

### Step 1: Fragmentise.py

python ${SCRIPTHOME}/Fragmentise.py --ligandfolder ${LigandSourceFolder} --datadir ${DATAHOME}

### Step 2: Run the ReduceFEATUREQueryVectors.ipynb.ipynb in the respective directories of EnsFragFeature. Take note of the number of FeatureVectors in the reduced file.

no_FeatureVectors=300

### Step 3: AnalyseGPU.py 

python ${SCRIPTHOME}/AnalyseGPU.py --datadir ${DATAHOME} --save ${ModelOut} --name ${MODELNUM}

#================================================================

# INFO

#================================================================


    # resatmlist = ['ALA.CB','ARG.CZ','ASN.PSEU','ASP.PSEU','CYS.SG','GLN.PSEU','GLU.PSEU',
    #               'HIS.PSEU','ILE.CB','LEU.CB','LYS.NZ','MET.SD','PHE.PSEU','PRO.PSEU','RES.N',
    #               'RES.O','SER.OG','THR.OG1','TRP.NE1','TRP.PSEU','TYR.OH','TYR.PSEU','VAL.CB']

    # IndividualResatmlist = ['RES.N', 'RES.O', 'ARG.CZ','HIS.PSEU','PRO.PSEU', 'TRP.NE1','LYS.NZ']
    # AliphaticResatmlist = ['ALA.CB', 'ILE.CB','LEU.CB','MET.SD', 'VAL.CB']
    # AromaticResatmlist = ['PHE.PSEU', 'TRP.PSEU', 'TYR.PSEU']
    # HydroxylResatmlist = ['CYS.SG', 'SER.OG','THR.OG1', 'TYR.OH']
    # AmideResatmlist = ['ASN.PSEU','GLN.PSEU']
    # CarboxylResatmlist = ['ASP.PSEU', 'GLU.PSEU']

    # ClassResatmlist={"IND.X": IndividualResatmlist, "ALI.CT": AliphaticResatmlist, "HYD.OH": HydroxylResatmlist, "ARO.PSEU": AromaticResatmlist, "CON.PSEU": AmideResatmlist, "COO.PSEU": CarboxylResatmlist}

