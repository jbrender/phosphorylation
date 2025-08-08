# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 00:18:57 2025

@author: jbren
"""

# =============================================================================
# USAGE AND DESCRIPTION
# =============================================================================
# This script generates structural and energetic features for a protein mutation
# and uses a pre-trained CatBoost model to predict a quantitative outcome.
#
# BEFORE RUNNING:
# 1. Set the paths to your trained CatBoost model and config file in the
#    "Prediction Model Paths" section below.
# 2. Ensure DSSP and FoldX are installed and their paths are correct.
#
# REQUIRED PYTHON PACKAGES:
# - pandas, numpy, biopython, catboost
#
# --- MODES OF OPERATION ---
#
# 1. single:
#    Processes a single PDB file and mutation, then prints the prediction.
#    Example (with full path):
#    python catboost_features.py single --pdb_file "path/to/protein.pdb" --mutation "S123A" --chain "A"
#    Example (with default PDB directory):
#    python catboost_features.py single --pdb_dir "path/to/pdbs" --pdb_file "protein.pdb" --mutation "S123A"
#
# 2. dataset:
#    Processes a CSV file containing 'pdb' and 'mutation' columns, predicts
#    the outcome for each row, and saves the results to a new CSV file.
#    The CSV can optionally contain a 'chain' column.
#    Example:
#    python catboost_features.py dataset tsu --output_csv "predictions.csv"
#
# =============================================================================

import os
import sys
import subprocess
import csv
import numpy as np
import pandas as pd
import argparse
import json
import re
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP

# =============================================================================
# 1. CONFIGURATION - PLEASE EDIT THESE PATHS
# =============================================================================
# --- Prediction Model Paths ---
MODEL_PATH = r"H:\My Drive\catboost_model99.cbm"
CONFIG_PATH = r"H:\My Drive\catboost_model_CONFIG99.json"

# --- Dataset Path Configurations ---
DATASET_PATHS = {
    'cancer': {
        'input_csv': r"C:\path\to\your\final_cancer.csv",
        'pdb_path': r'C:\path\to\your\pdb_cancer',
    },
    'tsu': {
        'input_csv': r"C:\path\to\your\tsui.csv",
        'pdb_path': r'C:\path\to\your\pdb_files',
    }
}

# --- Executable and Data Paths ---
FOLDX_EXE = r"C:\Users\jbren\Documents\FoldX\foldx_20251231.exe"

DSSP_EXE = r"C:\Users\jbren\Documents\Documents\phosphomutations\dssp-2.2.1-win64.exe"

if not os.path.exists(DSSP_EXE):
    print(f"FATAL ERROR: DSSP executable not found at the specified path.\nPlease check the DSSP_EXE path in the script.\nPath: {DSSP_EXE}")
    sys.exit(1)
if not os.path.exists(FOLDX_EXE):
    print(f"FATAL ERROR: FoldX executable not found at the specified path.\nPlease check the FOLDX_EXE path in the script.\nPath: {FOLDX_EXE}")
    sys.exit(1)

# --- FoldX Specific Paths ---
REPAIRED_PDB_PATH = './foldx_repair'
MUTANT_PDB_PATH = './Mutant_Structures'
INDIVIDUAL_LIST_FILENAME = 'individual_list.txt'

# --- Analysis Flags ---
FORCE_FOLDX_REPAIR = False #If TRUE force repair even if a repaired pdb exists
FORCE_FOLDX_BUILDMODEL = True

# --- Library Checks for Prediction ---
try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    print("FATAL ERROR: The 'catboost' library is not installed. Please run 'pip install catboost'.")
    sys.exit(1)


# =============================================================================
# 2. ATOM CLASSIFICATION DEFINITIONS
# =============================================================================
YRB_POSITIVE_CHARGE_SOURCES = {('LYS', 'NZ'), ('ARG', 'NE'), ('ARG', 'NH1'), ('ARG', 'NH2')}
YRB_NEGATIVE_CHARGE_SOURCES = {('ASP', 'OD1'), ('ASP', 'OD2'), ('GLU', 'OE1'), ('GLU', 'OE2')}
YRB_HYDROPHOBIC_CARBONS = {('ALA', 'CB'), ('ARG', 'CB'), ('ARG', 'CG'), ('ASN', 'CB'), ('ASP', 'CB'), ('CYS', 'CB'), ('GLN', 'CB'), ('GLN', 'CG'), ('GLU', 'CB'), ('GLU', 'CG'), ('HIS', 'CB'), ('HIS', 'CG'), ('HIS', 'CD2'), ('HIS', 'CE1'), ('ILE', 'CB'), ('ILE', 'CG1'), ('ILE', 'CG2'), ('ILE', 'CD1'), ('LEU', 'CB'), ('LEU', 'CG'), ('LEU', 'CD1'), ('LEU', 'CD2'), ('LYS', 'CB'), ('LYS', 'CG'), ('LYS', 'CD'), ('MET', 'CB'), ('MET', 'CG'), ('MET', 'CE'), ('PHE', 'CB'), ('PHE', 'CG'), ('PHE', 'CD1'), ('PHE', 'CD2'), ('PHE', 'CE1'), ('PHE', 'CE2'), ('PHE', 'CZ'), ('PRO', 'CB'), ('PRO', 'CG'), ('PRO', 'CD'), ('THR', 'CG2'), ('TRP', 'CB'), ('TRP', 'CG'), ('TRP', 'CD2'), ('TRP', 'CE3'), ('TRP', 'CZ2'), ('TRP', 'CZ3'), ('TRP', 'CH2'), ('TYR', 'CB'), ('TYR', 'CG'), ('TYR', 'CD1'), ('TYR', 'CD2'), ('TYR', 'CE1'), ('TYR', 'CE2'), ('TYR', 'CZ'), ('VAL', 'CB'), ('VAL', 'CG1'), ('VAL', 'CG2')}
SIDECHAIN_POLAR_ATOMS_SPECIFIC = {('ASN', 'OD1'), ('ASN', 'ND2'), ('CYS', 'SG'), ('GLN', 'OE1'), ('GLN', 'NE2'), ('HIS', 'ND1'), ('HIS', 'NE2'), ('MET', 'SD'), ('SER', 'OG'), ('THR', 'OG1'), ('TRP', 'NE1'), ('TYR', 'OH')}


# =============================================================================
# 3. Structural Features
# =============================================================================
class ProteinSecondaryStructureAnalyzer:
    def __init__(self, pdb_file, dssp_executable, chain='A'):
        self.chain = chain
        self.pdb_file = pdb_file
        if not os.path.exists(pdb_file):
            raise FileNotFoundError(f"PDB file not found: {pdb_file}")
        
        parser = PDBParser(QUIET=True)
        structure_id = os.path.basename(pdb_file).replace('.pdb', '')
        self.structure = parser.get_structure(structure_id, pdb_file)
        self.model = self.structure[0]
        self.dssp = DSSP(self.model, pdb_file, dssp=dssp_executable)
        self.position = None
        self.alt_position = None
        self.is_beta_hairpin = False
        self.ss = None
        self.RelSASA = np.nan

    def _get_ca_coordinates(self, residue_number):
        try:
            return self.model[self.chain][residue_number]['CA'].get_coord()
        except KeyError:
            return None

    def _is_current_sheet_hairpin(self, sheet_start, sheet_end):
        if sheet_end - sheet_start < 2: return False
        c_term_residues = [sheet_end - 2, sheet_end - 1, sheet_end]
        search_end = min(sheet_end + 20, sheet_end + 50)
        for search_res in range(sheet_end + 1, search_end):
            try:
                search_key = (self.chain, (' ', search_res, ' '))
                if self.dssp[search_key][2] in ['E', 'B']:
                    if self._check_hairpin_distance(c_term_residues, search_res):
                        return True
            except KeyError:
                continue
        return False

    def _check_hairpin_distance(self, c_term_residues, other_sheet_residue):
        other_coord = self._get_ca_coordinates(other_sheet_residue)
        if other_coord is None: return False
        for res_id in c_term_residues:
            c_term_coord = self._get_ca_coordinates(res_id)
            if c_term_coord is not None:
                if np.linalg.norm(c_term_coord - other_coord) <= 5.0:
                    return True
        return False

    def _is_n_terminal_hairpin_sheet(self, sheet_start, sheet_end):
        search_end = min(sheet_end + 50, sheet_end + 100)
        for search_res in range(sheet_end + 1, search_end):
            try:
                search_key = (self.chain, (' ', search_res, ' '))
                if self.dssp[search_key][2] in ['E', 'B']:
                    return True
            except KeyError:
                continue
        return False

    def analyze_residue(self, residue_number):
        residue_number = int(residue_number)
        try:
            dssp_key = (self.chain, (' ', residue_number, ' '))
            self.ss, self.RelSASA = self.dssp[dssp_key][2], self.dssp[dssp_key][3]
        except KeyError:
            return

        if self.ss == '-':
            self.is_beta_hairpin = False
            return

        if self.ss in ['H', 'G', 'I', 'T', 'S']:
            position, current_residue = 1, residue_number - 1
            while current_residue > 0:
                try:
                    if self.dssp[(self.chain, (' ', current_residue, ' '))][2] == self.ss:
                        position += 1
                        current_residue -= 1
                    else: break
                except KeyError: break
            self.is_beta_hairpin = False
            self.position = position
            self.alt_position = min(self.position - 1, 4)

        elif self.ss in ['E', 'B']:
            sheet_start, sheet_end = residue_number, residue_number
            res_cursor = residue_number - 1
            while res_cursor > 0:
                try:
                    if self.dssp[(self.chain, (' ', res_cursor, ' '))][2] in ['E', 'B']:
                        sheet_start = res_cursor
                        res_cursor -= 1
                    else: break
                except KeyError: break
            res_cursor = residue_number + 1
            while True:
                try:
                    if self.dssp[(self.chain, (' ', res_cursor, ' '))][2] in ['E', 'B']:
                        sheet_end = res_cursor
                        res_cursor += 1
                    else: break
                except KeyError: break
            
            self.is_beta_hairpin = self._is_current_sheet_hairpin(sheet_start, sheet_end)
            if not self.is_beta_hairpin:
                self.position = sheet_end - residue_number + 1
            else:
                if self._is_n_terminal_hairpin_sheet(sheet_start, sheet_end):
                    self.position = sheet_end - residue_number + 1
                else:
                    self.position = residue_number - sheet_start + 1

    def count_atoms_by_property_in_radius(self, target_res_num, radius):
        counts = {'hydrophobic_carbons': 0, 'positive_charge_source_atoms': 0,
                  'negative_charge_source_atoms': 0, 'polar_atoms': 0, 'total_atoms': 0}
        try:
            target_residue = self.model[self.chain][(' ', int(target_res_num), ' ')]
        except KeyError:
            return counts
        
        target_coords = [atom.get_coord() for atom in target_residue]
        if not target_coords: return counts
        
        center = np.mean(target_coords, axis=0)
        
        for residue in self.model.get_residues():
            if residue.id[0] != ' ' or residue.get_parent().id != self.chain: continue
            res_name = residue.get_resname()
            for atom in residue:
                if np.linalg.norm(atom.get_coord() - center) <= radius:
                    counts['total_atoms'] += 1
                    atom_id, atom_elem = atom.get_id()[0], atom.element.upper()
                    classified = False
                    if (res_name, atom_id) in YRB_POSITIVE_CHARGE_SOURCES:
                        counts['positive_charge_source_atoms'] += 1; classified = True
                    elif (res_name, atom_id) in YRB_NEGATIVE_CHARGE_SOURCES:
                        counts['negative_charge_source_atoms'] += 1; classified = True
                    elif atom_elem == 'C' and (res_name, atom_id) in YRB_HYDROPHOBIC_CARBONS:
                        counts['hydrophobic_carbons'] += 1; classified = True
                    
                    if not classified:
                        if atom_id in ('N', 'O') or (res_name, atom_id) in SIDECHAIN_POLAR_ATOMS_SPECIFIC:
                            counts['polar_atoms'] += 1
        return counts

# =============================================================================
#  FoldX
# =============================================================================
def run_foldx_command(command_string):
    """A helper to run a FoldX command and handle errors."""
    print(f"Running FoldX command: {command_string}")
    try:
        result = subprocess.run(command_string, shell=False, capture_output=True, text=True, check=True)
        print("FoldX ran successfully.")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: FoldX command failed.")
        print(f"Stderr: {e.stderr}")
        print(f"Stdout: {e.stdout}")
        raise
    except FileNotFoundError:
        print(f"ERROR: Could not find FoldX executable. Please check FOLDX_EXE path.")
        raise

def repair_pdb(pdb_id, pdb_path):
    if not os.path.isdir(REPAIRED_PDB_PATH):
        print(f"Creating directory for repaired PDBs: {REPAIRED_PDB_PATH}")
        os.makedirs(REPAIRED_PDB_PATH)
    repaired_pdb_file = f"{pdb_id}_Repair.pdb"
    cmd_repair = (f"{FOLDX_EXE} --command=RepairPDB --pdb-dir={pdb_path} --pdb={pdb_id}.pdb "
                  f"--output-dir={REPAIRED_PDB_PATH} --output-file={repaired_pdb_file} --water=IGNORE")
    run_foldx_command(cmd_repair)
    cmd_optimize = (f"{FOLDX_EXE} --command=Optimize --pdb-dir={REPAIRED_PDB_PATH} --pdb={repaired_pdb_file} "
                    f"--output-dir={REPAIRED_PDB_PATH} --water=IGNORE")
    run_foldx_command(cmd_optimize)
    print(f"Finished repairing and optimizing {pdb_id}")
    return f"Optimized_{repaired_pdb_file}"

def parse_foldx_output(dif_file_path):
    """Parses the FoldX Dif_ file to get energy terms."""
    if not os.path.exists(dif_file_path):
        print(f"Warning: FoldX output file not found: {dif_file_path}")
        return {}
    with open(dif_file_path, 'r') as file:
        lines = file.readlines()
    if len(lines) < 10:
        print(f"Warning: FoldX output file is malformed: {dif_file_path}")
        return {}
    header = [h.strip() for h in lines[8].split('\t')]
    values = [v.strip() for v in lines[9].split('\t')]
    foldx_results = {f"FoldX_{h}": v for h, v in zip(header, values)}
    return foldx_results

def build_model_and_analyze(repaired_pdb_filename, mutation_str, chain='A'):
    if not os.path.isdir(MUTANT_PDB_PATH):
        print(f"Creating directory for mutant PDBs: {MUTANT_PDB_PATH}")
        os.makedirs(MUTANT_PDB_PATH)
    
    # HIGHLIGHT: Check for multiple mutations and raise an error
    if ',' in mutation_str:
        raise ValueError("Multiple mutations are not supported. Please provide only one mutation at a time.")
    # END HIGHLIGHT
    
    original_res = mutation_str[0]
    if mutation_str[-1].isalpha():
        res_num = mutation_str[1:-1]
    else:
        res_num = mutation_str[1:]
    
    mutation_map = {'Y': 'y', 'S': 's', 'T': 'p'}
    new_res_code = mutation_map.get(original_res, 'a')
    
    foldx_mutation = f"{original_res}{chain}{res_num}{new_res_code};"
    with open(INDIVIDUAL_LIST_FILENAME, "w") as f:
        f.write(foldx_mutation)

    output_prefix = f"{os.path.splitext(repaired_pdb_filename)[0]}_{mutation_str}"
    cmd_build = (f"{FOLDX_EXE} --command=BuildModel --pdb-dir={REPAIRED_PDB_PATH} "
                 f"--pdb={repaired_pdb_filename} --mutant-file={INDIVIDUAL_LIST_FILENAME} "
                 f"--output-dir={MUTANT_PDB_PATH} --output-file={output_prefix} "
                 f"--water=IGNORE --pH=7.3 --ionStrength=0.15 --out-pdb=true")
    run_foldx_command(cmd_build)
    dif_file_path = os.path.join(MUTANT_PDB_PATH, f"Dif_{output_prefix}_{os.path.splitext(repaired_pdb_filename)[0]}.fxout")
    return parse_foldx_output(dif_file_path)

# =============================================================================
#  PREDICTION
# =============================================================================
def preprocess_data_for_model(df):
    if 'SS' in df.columns:
        ss_mapping = {'-': 'C', 'T': 'C', 'S': 'C', 'E': 'E', 'B': 'E', 'H': 'H', 'G': 'H', 'I': 'H'}
        df['SS'] = df['SS'].map(ss_mapping).fillna('C')
    return df

def get_prediction(feature_dict, model, config):
    df = pd.DataFrame([feature_dict])
    X_predict = preprocess_data_for_model(df)
    
    base_features = config.get('base_feature_cols', [])
    if not all(col in X_predict.columns for col in base_features):
        missing = [col for col in base_features if col not in X_predict.columns]
        print(f"Warning: Missing required features for prediction: {missing}. Skipping.")
        return np.nan
        
    return model.predict(X_predict[base_features])[0]

# =============================================================================
# 5. MAIN EXECUTION
# =============================================================================
def run_analysis(args):
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"FATAL ERROR: CatBoost model not found at the specified path.\nPlease check the MODEL_PATH in the script.\nPath: {MODEL_PATH}")
            sys.exit(1)
        if not os.path.exists(CONFIG_PATH):
            print(f"FATAL ERROR: Model config not found at the specified path.\nPlease check the CONFIG_PATH in the script.\nPath: {CONFIG_PATH}")
            sys.exit(1)
            
        with open(CONFIG_PATH, 'r') as f: config = json.load(f)
        model = CatBoostRegressor()
        model.load_model(MODEL_PATH)
        print(f"Loaded CatBoost model from {MODEL_PATH}")
    except Exception as e:
        sys.exit(f"FATAL ERROR: Could not load model. Error: {e}")

    # --- Setup DataFrame for Processing ---
    if args.mode == 'dataset':
        PDB_PATH = args.pdb_dir if args.pdb_dir else DATASET_PATHS[args.name]['pdb_path']
        input_csv_path = DATASET_PATHS[args.name]['input_csv']
        if not os.path.exists(input_csv_path):
            print(f"FATAL ERROR: The input CSV for the '{args.name}' dataset was not found.")
            print(f"Please check the path in the script: {input_csv_path}")
            sys.exit(1)
        df = pd.read_csv(input_csv_path)
    elif args.mode == 'single':
        if args.pdb_dir:
            PDB_PATH = args.pdb_dir
            pdb_id = args.pdb_file.replace('.pdb', '')
        else:
            PDB_PATH = os.path.dirname(args.pdb_file)
            pdb_id = os.path.basename(args.pdb_file).replace('.pdb', '')
        df = pd.DataFrame([{'pdb': pdb_id, 'mutation': args.mutation, 'chain': args.chain}])

    # --- Feature Generation and Prediction Loop ---
    all_results = []
    for index, row in df.iterrows():
        chain_for_row = row.get('chain', args.chain)
        print("-" * 50 + f"\nProcessing row {index+1}/{len(df)}: PDB={row['pdb']}, Mutation={row['mutation']}, Chain={chain_for_row}")
        current_row_results = {'pdb': row['pdb'], 'mutation': row['mutation'], 'chain': chain_for_row}
        
        try:
            # Part 1: DSSP and Atom Count Features
            pdb_file_path = os.path.join(PDB_PATH, f"{row['pdb']}.pdb")
            analyzer = ProteinSecondaryStructureAnalyzer(pdb_file_path, DSSP_EXE, chain=chain_for_row)
            
            mutation = row['mutation']
            if mutation[-1].isalpha():
                res_num = mutation[1:-1]
            else:
                res_num = mutation[1:]
            
            analyzer.analyze_residue(res_num)
            atom_10A = analyzer.count_atoms_by_property_in_radius(res_num, 10.0)

            current_row_results.update({
                'SS': analyzer.ss, 'RelSASA': analyzer.RelSASA, 'AltPosition': analyzer.alt_position,
                'total-long': atom_10A['total_atoms'], 'Hydrophobic-long': atom_10A['hydrophobic_carbons'], 
                'Polar-long': atom_10A['polar_atoms'], 'SASA': analyzer.RelSASA * 130 
            })
            
            # Part 2: FoldX Energy Features
            repaired_pdb_filename = f"Optimized_{row['pdb']}_Repair.pdb"
            repaired_pdb_full_path = os.path.join(REPAIRED_PDB_PATH, repaired_pdb_filename)

            if FORCE_FOLDX_REPAIR or not os.path.exists(repaired_pdb_full_path):
                print("  Running FoldX RepairPDB...")
                repaired_pdb_filename = repair_pdb(row['pdb'], PDB_PATH)
            else:
                print("  Skipping FoldX repair, using existing file.")
            
            if FORCE_FOLDX_BUILDMODEL:
                print("  Running FoldX BuildModel...")
                foldx_energy_terms = build_model_and_analyze(repaired_pdb_filename, row['mutation'], chain=chain_for_row)
                foldx_energy_terms['delta_total_energy']=foldx_energy_terms['FoldX_total energy']
                current_row_results.update(foldx_energy_terms)
                print(f"  FoldX analysis complete. Found {len(foldx_energy_terms)} energy terms.")
            
            # Part 3: Prediction
            prediction = get_prediction(current_row_results, model, config)
            current_row_results['prediction'] = prediction
            print(f"  -> Prediction: {prediction:.4f}")

        except Exception as e:
            print(f"  !! An error occurred: {e}")
            current_row_results['prediction'] = np.nan
        
        all_results.append(current_row_results)

    # --- Output Results ---
    if args.mode == 'single':
        if all_results and 'prediction' in all_results[0] and pd.notna(all_results[0]['prediction']):
            print("\n" + "="*25 + "\n      FINAL PREDICTION\n" + "="*25)
            print(f"  Value: {all_results[0]['prediction']:.4f}")
            print("="*25)
    elif args.mode == 'dataset':
        output_df = pd.DataFrame(all_results)[['pdb', 'mutation', 'chain', 'prediction']]
        output_df.to_csv(args.output_csv, index=False)
        print(f"\nProcessing complete. Predictions saved to: {args.output_csv}")
		
def main():
    parser = argparse.ArgumentParser(description="Run protein feature generation and prediction with CatBoost.",
                                     formatter_class=argparse.RawTextHelpFormatter)
    subparsers = parser.add_subparsers(dest='mode', title='Operating Modes',
                                       help="Choose one of the following modes:")

    # --- Parser for 'dataset' mode ---
    parser_dataset = subparsers.add_parser('dataset', help='Process a dataset CSV to generate features and predict.')
    parser_dataset.add_argument('name', choices=DATASET_PATHS.keys(), help=f"Dataset to process: {list(DATASET_PATHS.keys())}")
    parser_dataset.add_argument('--output_csv', required=True, help='Path for the output CSV file with predictions.')
    parser_dataset.add_argument('--chain', type=str, default='A', help='Default PDB chain ID if not in CSV (default: A).')
    parser_dataset.add_argument('--pdb_dir', type=str, help='Optional: Directory containing PDB files. Overrides default path.')

    # --- Parser for 'single' mode ---
    parser_single = subparsers.add_parser('single', help='Process a single PDB file to generate features and predict.')
    parser_single.add_argument('--pdb_file', required=True, help='Full path to the PDB file, or just the PDB ID if --pdb_dir is used.')
    parser_single.add_argument('--mutation', required=True, help="Mutation to analyze, e.g., 'S123A'.")
    parser_single.add_argument('--chain', type=str, default='A', help='The PDB chain ID to analyze (default: A).')
    parser_single.add_argument('--pdb_dir', type=str, help='Optional: Directory containing the PDB file.')
    
    args = parser.parse_args()

    if args.mode is None:
        parser.print_help(sys.stderr)
        sys.stderr.write("\nFATAL ERROR: You must specify an operating mode ('single' or 'dataset').\n")
        sys.exit(1)
    run_analysis(args)
# =============================================================================
# For debugging purposes only
# =============================================================================

# # class MockArgs:
# #     mode = 'single'
# #     pdb_file = r"C:\Users\jbren\Documents\Documents\phosphomutations\protein\structural_information\pdb_files\1A32.pdb" # replace
# #     mutation = "Y68"
# #     chain = "A"
# #     pdb_dir = None
# 
# 
# # run_analysis(MockArgs())
# =============================================================================

if __name__ == "__main__":
    main()