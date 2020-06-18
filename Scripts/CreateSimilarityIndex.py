"""
Creates indices of Target Testing/Validation Protein (i.e. HIV-1 Protease)
CHANGE PDBID Names in Line 22 and 25
"""

__author__ = "Michael Suarez"
__email__ = "masv@connect.ust.hk"
__copyright__ = "Copyright 2019, Hong Kong University of Science and Technology"
__license__ = "3-clause BSD"

from argparse import ArgumentParser
import numpy as np
import pickle

parser = ArgumentParser(description="Build Files")
parser.add_argument("--datadir", type=str, default="Data", help="input - XXX.YYY ")
parser.add_argument("--envNewAcronym", type=str, default="PRT.SNW", help="input - XXX.YYY ")

args = parser.parse_args()

# Protein PDBIDs 100% Matches of Testing/Validation Protein (i.e.HIV-1 Protease)
comp = ['1HTE','1HTF','1HTG','1XL2','1NPA','1NH0','1XL5','1OHR','1HSG','1HBV','1MUI','2FDE','3CKT','2PWC','2UXZ','4FE6','1YT9','3BHE','1SP5','2CEM','2PWR','2CEN','2AQU','2CEJ','1W5W','1EBW','1W5X','1W5Y','1C70','1EBZ','1EC0','1EC1','1EC2','3BGC','1EC3','3BGB','1W5V','2BB9','2WKZ','1DIF','3PSU','2BBB','2UPJ','2UY0','1HIH','1ZSR','3GGX','6D0D','3GGV','6D0E','3T11','1IIQ','2QNP','2QNQ','4I8Z','1HPS','4I8W','1AJV','1HPV','1AJX','1HHP','1HPX','2BQV','7UPJ','1GNO','1T7K','2ZGA','1M0B','3GGA','3PHV','2BPX','2BPY','2BPZ','1HOS','1ZSF','2QNN','1U8G','2BPV','2BPW','1HVI','1HVJ','1HVK','1HVL','2A4F','1HVC','1UPJ','1D4I','1D4J','1WBK','1G2K','1NPV','1NPW','4HLA','1D4H','2PQZ','1G35','1FQX','1WBM','5TYR','5TYS']

# Protein PDBIDs 70% sequence homology Matches of Testing/Validation Protein (i.e. 70% sim of HIV-1 Protease)
comp2 = ['1OHR','3BHE','1SP5','1G6L','3BGC','3BGB','1HXB','1ODW','1HVI','1HVJ','1HVK','1HVL','1HVC','1UPJ','3BC4','1G2K','1G35','1FQX','1HTE','1HTF','1HTG','1HSG','1YT9','2CEM','2CEN','2CEJ','1DIF','1HPS','1HPV','1HPX','1HOS','5TYR','5TYS','1HIV','3PSU','1HIH','1HHP','2ZGA','1M0B','2A4F','1WBK','1NPV','1NPW','1WBM','1NPA','2R3T','2R3W','1HBV','2R38','2R43','2UXZ','2UY0','2QNP','2QNQ','4I8Z','4I8W','2BQV','3PHV','2BPX','2BPY','2BPZ','2QNN','1U8G','2BPV','2BPW','1D4I','1D4J','1D4H','1NH0','3CKT','1W5W','1W5X','1W5Y','1W5V','2UPJ','1AJV','1AJX','4HLA','4FE6','2BB9','2WKZ','2BBB','1ZSR','3GGX','3GGV','1GNO','3GGA','1ZSF','1XL2','1XL5','1MUI','2FDE','2PWC','2PWR','1EBW','1EBZ','1EC0','1EC1','1EC2','1EC3','6D0D','6D0E','1IIQ','7UPJ','2PQZ','2AQU','1C70','3T11','1T7K','1MER','1MES','1HXW','3M9F','1HVS','1SBG','6MCR','6MCS','6OGP','3S85','2P3B','1AXA','1PRO','5IVQ','5IVR','5IVS','2QMP','3TLH','2HVP','1IZH','1AAQ','9HVP','1VIJ','1VIK','1TCX','1ZPA','4LL3','1ZP8','2O4P','2O4S','2O4K','1EBY','1A8G','5HVP','4PHV','1BV9','1MET','5COO','5COP','5CON','5COK','1D4S','2QHC','2Z54','2O4L','1BV7','1MEU','4EJD','6OXU','6OXV','6OXS','6OXT','6OXQ','6OY2','6OXR','4EJK','6OXO','6OY0','6OXP','6OY1','6OXY','6OXZ','6OXW','6OXX','3QBF','1ODX','1ODY','2PC0','1LZQ','6OPS','6OOS','1HPO','4GB2','3NLS','1HEF','1HEG','4E43','4MC1','4MC9','4MC6','4MC2','5IVT','6DGY','6DGZ','6DH0','6DH1','6DH8','6DGX','6DH6','6DH7','6DH4','6DH5','6DH2','6DH3','1D4Y','3TH9','5VEA','5VCK','2FLE','1ZTZ','1GNM','1GNN','6B3G','6B3H','6B36','6B38','6B3F','6B3C','1ZPK','2AZ8','1MRW','1ZLF','1MSM','4U7V','1A9M','3KFS','3KFR','3KFP','3KFN','1A8K','4TVH','4TVG','4U7Q','3KF0','3QRS','3QRO','3QRM','1ZJ7','2HB4','3KDC','3KDB','3QPJ','3KDD','3QP0','3QN8','1BWA','1BWB','1A30','1ZBG','2PK5','2PK6','3I8W','3QIH','5UFZ','1HWR','1HVH','1HVR','1DMP','1QBR','1QBS','1QBT','1QBU','6OOU','1LV1','4QGI','1RL8','2I4D','2I4U','2I4W','2WHH','2AZ9','2Q64','2NPH','1BVE','1BVG','5YRS','3DOX','2XYF','2XYE','6OPT','6OOT','4A6C','4A6B','4A4Q','1FB7','5AHB','5AHC','5AHA','5AGZ','5AH8','5AH9','5AH6','5AH7','1IZI','2I4V','2I4X','2QAK','3ZPU','3ZPT','3ZPS','2WL0','1MRX','1MSN','2Q63','4CPX','4CPW','4CPU','4CPT','4CPS','4CPR','4CPQ','4COE','4CP7','1Z8C','4QLH','3KT2','3KT5','4Q5M','2PYM','2PYN','1AID','5KAO','5V4Y','3I7E','2F3K','2AID','3O9D','3O9C','3O9B','3O9A','3SAC','3SAB','3SAA','3O9I','3O9H','3O9G','3O9F','3O9E','1YTG','1YTH','3SA8','3SA7','3SA6','3SA5','3SA4','3SA3','3O99','3SA9','3AID','4DJQ','4DJP','4DJO','4DJR','2QI3','2QI4','2QI5','2QI6','2QI7','2QI0','2QI1','2QHY','2QHZ','3EKY','4K4R','4K4Q','4K4P','3EL1','3EKX','3EKV','3MXE','3MXD','1KZK','2I0A','2I0D','3GI6','3GI5','3GI4','3N3I','4Q1X','2AZB','2PSU','2PSV','3R4B','2Q5K','2Q54','2Q55','2Q3K','4F74','4F73','4F76','4F75','1T3R','1K6C','1K6T','1K6V','1K6P','3EL5','3OY4','1MTB','4Q1W','2O4N','1T7I','1T7J','5WLO','3I6O','5DGW','5DGU','5YOK','3DK1','2ZYE','3DJK','3QAA','3K4V','6BZ2','4ZLS','6C8X','6OPU','5JFU','4ZIP','1SDT','6E9A','5JG1','5JFP','3NU3','6E7J','6DV4','6DV0','1FGC','3B7V','2IEN','4KB9','2A1E','6DIF','6DJ1','3NDX','3NDW','3NDU','3NDT','4OBK','4OBJ','4OBH','3TOH','3TOG','3TOF','3H5B','1TSU','1F7A','3TL9','3TKW','3TKG','4FL8','3EM3','4DFG','3EL9','3EL4','2Z4O','2FNS','6O48','6B4N','3OXC','3OXX','3OXW','3OXV','2NXD','2FGU','2FGV','2NXL','2NXM','6U7O','6U7P','4Q1Y','5BS4','5BRY','4JEC','4U8W','3A2O','2HB3','3ST5','1KJ4','1KJ7','1KJH','6IXD','1KJF','1KJG','5UPZ','3OK9','5UOV','3FX5','2AOI','2AOJ','2AOD','5ULT','6CDL','6CDJ','4EJL','3LZS','4EJ8','6OTG','1BDL','1BDR','3NUO','3NUJ','1SDU','3NU9','3NU6','3NU5','3NU4','3PWM','3PWR','3B80','3CYX','1FEJ','2IEO','3S56','5W5W','3S54','3S53','3S43','4QJ9','4QJA','3D20','6DJ2','6DIL','3D1Z','6DJ7','3D1Y','3D1X','6DJ5','5VJ3','4FLG','4FM6','3EM6','3EM4','2QD6','2QD8','2HS1','2HS2','3VF7','3VF5','4HDP','4HE9','5T8H','5E5K','5E5J','2AVO','2AVQ','2AVS','2AVM','2B60','2NNK','2NNP','2F8G','2AOC','2F81','1DW6','3LZV','3JW2','3JVY','3JVW','6C8Y','1SGU','1SH9','1BDQ','1SDV','6BRA','1FG6','1FG8','1FFI','1FFF','2P3A','2IDW','4QJ2','4QJ8','4QJ7','4QJ6','1TSQ','2G69','2QCI','2FNT','2QD7','3VFB','3VFA','4HEG','4HDF','4HDB','1MT8','1MT9','2AZC','1N49','1MT7','1EBK','1A94','2AVV','3BVB','3BVA','2HC0','2NMY','2NMZ','2HB2','2AOH','2AOE','2AOF','2AOG','2F80','3LZU','1K2B','1DAZ','1FF0','3U71','3D3T','5KR0','5KQY','5KQZ','5KQX','4RVX','4RVJ','4NJU','4NJS','2B7Z','3HVP','4L1A','1K2C','1K1U','8HVP','6OGR','6OGS','6OGQ','3CYW','6OGV','2R8N','2R5P','2R5Q','1B6J','4DQB','1RV7','4HVP','2FXE','1TW7','4OBG','1RQ9','4OBF','4OBD','3IXO','1RPI','1CPI','3EKP','3EL0','3EKW','3EKT','3EKQ','4FAF','4FAE','3OTY','3OUD','3OUC','3OUB','3OUA','3OU4','3OTS','5T84','3OU3','3OU1','3OQ7','4EYR','7HVP','3R0W','3R0Y','1C6X','1C6Y','1C6Z','6OPV','3PJ6','3OQD','3OQA','4NKK','2JE4','2RKF','5YOJ','6OPX','6MK9','6MKL','1Z1R','1Z1H','1K1T','6OGT','6OGL','2P3C','1B6M','1B6P','1B6K','1B6L','1D4K','1D4L','4M8X','4RVI','3BXS','1MTR','4GZF','4GYE','3SPK','3SO9','6OPW','5KR2','5KR1','4NJV','4NJT','2J9J','2J9K','6OPY','1Q9P','4DQF','6I45','2FXD','4YOA','4YOB','2FDD','3BXR','4EPJ','4EP3','4EP2','6OPZ','4DQH','4DQG','4DQE','4DQC','4EQJ','4EQ0','2RKG','3I2L','3DCR','3NXE','3NXN','3HLO','3HAU','4M8Y','2O40','3IAW','3KA2','3HZC','3NWQ','3NWX','3DCK','3IA9','3NYG','3HDK','3HBO','3HAW','3GI0','3FSM','4NPT','3U7S','3TTP','4NPU','3GGU','6O5A','6O5X','6O57','5T2Z','5T2E','5B18','4J5J','4J55','4J54','3UHL','3UFN','3UF3','3UCB','2P3D','6O54','6PRF','4YHQ','4YE3','4Z50','4Z4X','3T3C','6P9A','6P9B','3MWS']


# collects the environment names in order
coll = []
with open("../%s/%s/%s.Homogenised.annotation.txt" %(args.datadir, args.envNewAcronym, args.envNewAcronym), "r") as f:
    for line in f:
        coll.append(line.split()[0][4:8]) # Items are in Env_6ULC_0 Format with the center part corresponding to the PDBIds

print('Environments in Dataset: %s' %(len(coll)))

print('Target Protein Items: %s' %(len(comp)))
print('Target Protein Homology Items: %s' %(len(comp)))

# collects the environments exact same to the comp array by index
sums2 = []
for j in comp:
    if j.lower() in coll:
        sums2.append([i for i, e in enumerate(coll) if e == j.lower()])
indx_list2 = [item for sublist in sums2 for item in sublist]

print('Same within the Dataset: %s' %(len(indx_list2)))
pickle.dump(indx_list2, open('../%s/%s/%s.TargetProteinIndx.mtr' %(args.datadir, args.envNewAcronym, args.envNewAcronym), "wb"))


# collects the environments exact same to the comp array by index
sums = []
for j in comp2:
    if j.lower() in coll:
        sums.append([i for i, e in enumerate(coll) if e == j.lower()])
indx_list = [item for sublist in sums for item in sublist]

print('70% Homology within the Dataset: %s' %(len(indx_list)))
pickle.dump(indx_list, open('../%s/%s/%s.TargetProteinIndx_70.mtr' %(args.datadir, args.envNewAcronym, args.envNewAcronym), "wb"))



