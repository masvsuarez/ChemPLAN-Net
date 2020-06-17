## FragFeatureNet

FragFeatureNet relies on previous work done in the [Xuhui Huang Research group](http://compbio.ust.hk/public_html/pmwiki-2.2.8/pmwiki.php?n=Main.HomePage), the [FEATURE framework](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2559884/) by Russ B Altman and the [NucleicNet](https://www.nature.com/articles/s41467-019-12920-0) framework by Jordy H Lam.

### Getting started

Download the repository and install relevant python dependencies.

- Open the `commandfile.sh` and run the relevant scripts

### Outline of the FragFeatureNet pipeline

#### Training a new instance of the FragFeatureNet given the FEATURE output files on a protein family

1. 

#### Testing a pre-trained model of FragFeatureNet given the FEATURE vectors of the query environments and `.sdf` files of query ligands


### Code and data

#### `data/` directory



1. Start with fusing the data into one file using Fusedata.py
2. Remove Fragments which are not present in the FragKB
3. Create HIV1 Index
4. Create Non Binding
5. Train Data
