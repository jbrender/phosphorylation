Using DDG Phosphorylation
1. Download FoldX from the FoldX website
2. Download and compile DSSP from the DSSP github or use a pre-compiled version
3. Download ddGphosphorylation.py, catboost_model99.cbm, and catboost_model_CONFIG99.json from this site
3. Edit ddGphosphorylation.py to contain the locations of the downloaded programs and scripts from steps 2 and 3
Running on a single protein:
%run ddgPhosphorylation.py single --pdb_file <pdb_file_path) --mutation <mutation> --chain <chain code>
i.e %run ddgPhosphorylation.py single --pdb_file "C:\Users\jbren\Documents\pdb_files\1A32.pdb" --mutation "Y68" --chain "A"

Pre-Machine Learning
--------------------

prepare_tsu1.m
	Creates a list of PDB IDs, energies, and residues for the Tsuboyama data
	input: dG_non_redundant_natural_Fig5.csv
	output: phoslist_tsu.csv

prepare_tsu2.m
	Aligns sequences for Tsuboyama proteins and adds new residue numbers to list
	input: phoslist_tsu.csv
	output: phoslist_tsu_new.csv

prepare_pancancer1.m
	Creates a list of Uniprot IDs and residues for the cancer dataset
	input: mmc1.xlsx, mmc2.xlsx
 	output: phostable.csv

prepare_pancancer2.m
	Reads alignments for cancer proteins and adds new residue numbers to list
	input: phostable.csv, aligned sequences
	output: phoslist_with_conversion.csv

catboost_features.py
	Calculates FoldX and structural features
	Usage: python catboost_features.py tsu to calculate the features for the Tsuboyama dataset
	       python catboost_features.py cancer to calculate the features for the cancer dataset
		python your_script_name.py single --pdb_file <pdbfile> --mutation <ex. "S123A"> --output_csv <outfile> 
Machine Learning
--------------------
CatBoostRegression.py
	Reads the feature file from the external and Tsuiboyama datasets and constructs the cataboost model
	input: external and tsuiboyama datasets as csv
	ouutput: Catboost model 

Post-Machine Learning
---------------------
remove_duplicates.m
	Removes rows with the same uniprot ID and mutated residue number
	input: Dataset_6.csv (from machine learning)
	output: edited Dataset_6.csv

centralities.m
	Determines whether network locations with certain centrality measures are prone to harbor destabilizing phosphorylations
	input: Dataset_6.csv, mmc2.xlsx
 	output: box plots and p values for centralities with low and high ddG

search_humsavar.m
	Searches the humsavar database for tyrosine to aspartate mutations at the same residue positions as phosphorylations from the cancer dataset
	input: humsavar2.txt, Dataset_6.csv
	output: rows from cancer output data matching to humsavar

search_clinvar.m
	Searches the clinvar database for tyrosine to aspartate mutations at the same residue positions as phosphorylations from the cancer dataset
	input: variant_summary.txt, Dataset_6.csv
	output: rows from cancer output data matching to clinvar

find_cancer_rapid.m 
	Search for energy of mutation within all alphafold2 mutations, for the cancer ptm dataset
	input: alpha_pedictions.csv, Dataset_6.csv
	output: column of energies

psp_search.m
	Search phophosite plus for energy of mutation within alphafold2 mutations
	input: alpha_predictions.csv, posit
	output: energies

The list of all 300k tyrosine predictions from alphafold2 are given here: https://drive.google.com/file/d/1Tb6tZ4-Zc7kxQhbsGuGvEgLBZ_PjhNZ5/view?usp=sharing
	
