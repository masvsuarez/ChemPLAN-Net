##############################################################################
#
# FragFeatureNet
#                                                                     
##############################################################################

__author__ = "Michael Suarez"
__email__ = "masv@connect.ust.hk"
__copyright__ = "Copyright 2019, Hong Kong University of Science and Technology"
__license__ = "3-clause BSD"

SCRIPTHOME='Scripts'		# This is the default location of scripts
FEATURESPROT='../BinaryModel/DataP_New/PRT.SNW/PRT.SNW.Homogenised.property.pvar'		# This is the default location of the Data
DATAHOME='Data'				# This is the pre-processed Data home
MODELHOME='FNN'				# This is the default location of models
INQUIRY='Results_test_CDK2'	# This is the inquiry folder
PROTEINFAMILYACRONYM = 'PRT.SNW'

#Modeloutput
ModelOut="Results_Prot_New"

#Modelname
MODELNUM="BinaryModel_New00_FULL"


### Step 1: FuseData.py
# Environments to Fuse
#{"ALI.CT", "ARG.CZ", "ARO.PSEU", "CON.PSEU", "COO.PSEU", "HIS.PSEU", "HYD.OH", "LYS.NZ", "PRO.PSEU", "RES.N", "RES.O", "TRP.NE1"}

python ${SCRIPTHOME}/FuseData.py --datadir ${DATAHOME} --envNewAcronym ${PROTEINFAMILYACRONYM}

### Step 2: ReduceFragments.py

python ${SCRIPTHOME}/ReduceFragments.py --datadir ${DATAHOME} --envNewAcronym ${PROTEINFAMILYACRONYM}



#===================
# Train BinaryModel for Protease
#===================

srun --nodelist=node-2 nohup python ${SCRIPTHOME}/${MODELHOME}/trainOpt2_New_FULL.py ${FEATURESPROT} ${DATAHOME} --save ${ModelOut} --depth 65  -b 512 -lr 0.3 -m 0.01 -d 0.0001 -e 20 --name ${MODELNUM} --ngpu 8 > ${ModelOut}/${MODELNUM}.out &

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

