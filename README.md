# MDeePred: Multi-Channel Deep Chemogenomic Modeling of Receptor-Ligand Binding Affinity Prediction for Drug Discovery

![alt text](./figures/mdeepred_network_structure_figure.png)

## Protein Representation
![alt text](./figures/encoding_figure_mod.png)

## Descriptions of folders and files in the MDeePred repository

* **bin** folder includes the source code of MDeePred.

* **training_files** folder contains various traininig/test datasets mostly formatted for observational purposes and for employment in future studies
    * **Davis** contains training data points, features, data splits for Davis dataset based on Setting-1. **compound_feature_vectors/ecfp4_normalized.tsv** includes ecfp4 features of ligands. **dti_datasets/comp_targ_affinity.csv** is a csv file where each line is formatted as <compound_id>,<target_id>,<bioactivity_value>. **helper_files** includes smiles strings and fasta files of ligands and proteins, respectively. **target_feature_vectors** is for storing target feature vectors which can be downloaded from the following link.
    * **Davis_Filtered** contains training data points, features, data splits for Filtered Davis dataset based on Setting-1. **compound_feature_vectors/ecfp4_normalized.tsv** includes ecfp4 features of ligands. **dti_datasets/comp_targ_affinity.csv** is a csv file where each line is formatted as <compound_id>,<target_id>,<bioactivity_value>. **helper_files** includes smiles strings and fasta files of ligands and proteins, respectively. **target_feature_vectors** is for storing target feature vectors which can be downloaded from the following link.
    * **PDBBind_Refined** contains training data points, features, data splits for PDBBind Refined dataset based on Setting-2. **compound_feature_vectors/ecfp4_normalized.tsv** includes ecfp4 features of ligands. **dti_datasets/comp_targ_affinity.csv** is a csv file where each line is formatted as <compound_id>,<target_id>,<bioactivity_value>. **helper_files** includes smiles strings and fasta files of ligands and proteins, respectively. **target_feature_vectors** is for storing target feature vectors which can be downloaded from the following link.
    * **kinome** contains training data points, features, data splits for Kinase dataset based on Setting-2. **compound_feature_vectors/ecfp4_normalized.tsv** includes ecfp4 features of ligands. **dti_datasets/comp_targ_affinity.csv** is a csv file where each line is formatted as <compound_id>,<target_id>,<bioactivity_value>. **helper_files** includes smiles strings and fasta files of ligands and proteins, respectively. **target_feature_vectors** is for storing target feature vectors which can be downloaded from the following link.

## Development and Dependencies

#### [PyTorch 0.4.1](https://pytorch.org/get-started/previous-versions/)
#### [Pandas 0.23.4](https://pandas.pydata.org/pandas-docs/version/0.23.4/install.html)
#### [Sklearn 0.20](https://scikit-learn.org/0.20/install.html)
#### [Numpy 1.14.5](https://pypi.python.org/pypi/numpy/1.13.3)


## How to run pre-trained ready-to-use MDeePred models for binding affinity predictions

**Output of the script:**


## How to train a model

**Output of the script:**




## How to re-produce performance comparison results for MDeePred and other methods 

```
python main_training --chln 1024_1024 --tlnaf 256 --lhln 1024_1024 --lr 0.0001 --bs 32 --td Davis --cf ecfp4 --tf sequencematrix500_ZHAC000103LEQ500_GRAR740104LEQ500_SIMK990101LEQ500_blosum62LEQ500 --setting 1 --dropout 0.1 --en davis_dataset_retraining
```

```
python main_training --chln 1024_1024 --tlnaf 128 --lhln 1024_512 --lr 0.0001 --bs 32 --td Davis_Filtered --cf ecfp4 --tf sequencematrix500_ZHAC000103LEQ500_GRAR740104LEQ500_SIMK990101LEQ500_blosum62LEQ500  --setting 1 --dropout 0.1 --en davis_filtered_dataset_retraining
```


```
python main_training --chln 1024_1024 --tlnaf 128 --lhln 1024_512 --lr 0.0001 --bs 32 --td Davis_Filtered --cf ecfp4 --tf sequencematrix500_ZHAC000103LEQ500_GRAR740104LEQ500_SIMK990101LEQ500_blosum62LEQ500  --setting 2 --dropout 0.25 --en pdbbind_refined_dataset_retraining
```


## License

    Copyright (C) 2019 CanSyL

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.

