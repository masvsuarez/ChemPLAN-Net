## FragFeatureNet

FragFeatureNet relies on previous work done in the [Xuhui Huang Research group](http://compbio.ust.hk/public_html/pmwiki-2.2.8/pmwiki.php?n=Main.HomePage), the [FEATURE framework](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2559884/) by Russ B Altman and the [NucleicNet](https://www.nature.com/articles/s41467-019-12920-0) framework by Jordy H Lam.

### Getting started

Download the repository and install relevant python dependencies.

- Open the `commandfile.sh` and run the relevant scripts

### Outline of the FragFeatureNet pipeline

#### Training a new instance of the FragFeatureNet 
...given the FEATURE output files on a protein family

1. Save the individual Environment `annotation.txt` (PDB-ID and Environment Location Annotation), `boundfrags.txt` (Binding Fragment CIDs) and `property.pvar` (Environment Feature Vectors in binary file) FEATURE Data files in the `Data` directory. Make sure they are saved in the format of `Data/Env/Env.file`, where `file` corresponds to one of the three file names above and `Env` corresponds to one each of the following: `ALI.CT, ARG.CZ, ARO.PSEU, CON.PSEU, COO.PSEU, HIS.PSEU, HYD.OH, LYS.NZ, PRO.PSEU, RES.N, RES.O, TRP.NE1`. The `FuseData.py` Script has to be altered if not all environments are present. Save the Fragment-base Dictionary as `GrandCID.dict` (Pandas Dictionary) in the `Data` Directory.

E.g.
```
.
├── `Data/`
│   ├── `GrandCID.dict`
│   ├── `ALI.CT/`
│   │   ├── `ALI.CT.annotation.txt`
│   │   ├── `ALI.CT.boundfrags.txt`
│   │   └── `ALI.CT.property.pvar`
│   ├── `ARO.PSEU/`

```

2. Run `FuseData.py` to merge the files of the environments into three large protein family files. Select an acronym for the merged files, i.e. `PRT.SNW` for Proteases New, or `KIN.ALL` for All Kinases.

3. Run `ZeropaddingBoundfrags.py` to re-format the binding fragments. TODO: Non-Ideal storage size or format.

4. Environments that contain Fragments not in the original Fragment Database have to be removed. The merged `annotation.txt`, `boundfrags.txt` and `property.pvar` files will be permanently altered, so make a copy before executing this script. Run `ReduceFragments.py`. Now your Environments, Fragments and Annotations are ready for pre-processing.

5. Find the Indices of the Protein of your choice through the annotation file for removal or specialised training/validation. Make sure you substitute the relevant PDBIDs in the `CreateSimilarityIndex.py` Script in the List of Strings format. Two Lists will be required: Firstly, PDBIDs of Identical Proteins to the Test/Validation Protein and secondly, PDBIDs of 70% Sequence Homolog Proteins to the Test/Validation Protein. Run `CreateSimilarityIndex.py`.

6. Create Non Binding Data in form of indicies in the `GrandCID.dict` Dataframe. Non Binding data is based on the pairwise dissimilarity of binding and nonbinding Fragments. Run `CreateNonBinding.py`.

7. Prepare Data into Format required for the Training of the Network. Run `DataPrep.py`.

#### Training the network

Run `FNN/train_test_model.py` with adequate hyperparameters on the training and testing data. If you want to train the model on the complete data and conduct the queries afterwards run `FNN/train_model.py`. 


#### Testing a pre-trained model of FragFeatureNet

1. Save the Ligands to be cross-referenced as in a `Ligands` directory as `.sdf` files of query ligands. Run `Fragmentise.py`. You might want to change the number of parallel processes on line 93 depending on your systems. 

2. Save your `AllQuery.df` containing the query Feature Vectors in the `Data` directory. This Dataframe is obtained by the EnsFragFeature code as output of fpocket given the query PDB structures. TODO: Implement this here as well. Make sure you reduced the Query Vectors by running the `ReduceFEATUREQueryVectors.ipynb` in your respective EnsFragFeature directory.

3. Run `AnalyseGPU.py` to obtain the binding probabilities in pickled `.mat` files.


### Code and data

#### `data/` directory


### Contacts
If you have any questions or comments, please feel free to email Michael Suarez (masv[at]connect[dot]ust[dot]com).
