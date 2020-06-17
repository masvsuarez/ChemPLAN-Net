## FragFeatureNet

FragFeatureNet relies on previous work done in the [Xuhui Huang Research group](http://compbio.ust.hk/public_html/pmwiki-2.2.8/pmwiki.php?n=Main.HomePage), the [FEATURE framework](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2559884/) by Russ B Altman and the [NucleicNet](https://www.nature.com/articles/s41467-019-12920-0) framework by Jordy H Lam.

### Getting started

Download the repository and install relevant python dependencies.

- Open the `commandfile.sh` and run the relevant scripts

### Outline of the FragFeatureNet pipeline

#### Training a new instance of the FragFeatureNet 
...given the FEATURE output files on a protein family

1. Save the individual Environment `annotation.txt`, `boundfrags.txt` and `property.pvar` FEATURE Data files in the `Data` directory. Make sure they are saved in the format of `Data/Env/Env.file`, where `file` corresponds to one of the three file names above and `Env` corresponds to one each of the following: `ALI.CT, ARG.CZ, ARO.PSEU, CON.PSEU, COO.PSEU, HIS.PSEU, HYD.OH, LYS.NZ, PRO.PSEU, RES.N, RES.O, TRP.NE1`. `fusedata.py` Script has to be altered if not all environments are present.

.
+--`data/`
|   +-- `ALI.CT/`
|   |   +-- `ALI.CT.annotation.txt`
|   |   +-- `ALI.CT.boundfrags.txt`
|   |   +-- `ALI.CT.property.pvar`

2. Run `fusedata.py` to merge the files of the environments into three large protein family files. Select an acronym for the merged files, i.e. `PRT.SNW` for Proteases New, or `KIN.ALL` for All Kinases.




#### Testing a pre-trained model of FragFeatureNet 
...given the FEATURE vectors of the query protein environments and `.sdf` files of query ligands


### Code and data

#### `data/` directory



1. Start with fusing the data into one file using Fusedata.py
2. Remove Fragments which are not present in the FragKB
3. Create HIV1 Index
4. Create Non Binding
5. Train Data

### Contacts
If you have any questions or comments, please feel free to email Michael Suarez (masv[at]connect[dot]ust[dot]com).
