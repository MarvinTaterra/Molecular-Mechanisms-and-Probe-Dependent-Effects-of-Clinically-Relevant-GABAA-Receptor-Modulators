"""
Extract Selected Distances for PLUMED CV Input

Inputs: original PLUMED distance file (all ~1.4M distances),
the selected_distances.txt file (10,000 selected distances)

Output: A new PLUMED file with only the selected distances in the correct order,
Maintains atom numbers and comments from original file

"""

import re
from pathlib import Path
import numpy as np



CONFIG = {
    # Input files
    'original_plumed': 'path/to/1/distances.dat', 
    'selected_distances': 'path/to/selected_distances.txt',  
    
    # Output file
    'output_plumed': 'path/to/Phenobarbital_GABA_torch.dat', 
    
    'create_summary': False,
    'summary_file': 'path/to/distance_extraction_summary.txt',
}



def parse_selected_distances(filepath):
    """
    Parse selected_distances.txt to get the list of d-numbers in order.
    
    Returns:
        list: Ordered list of d-numbers (e.g., [12345, 67890, ...])
    """
    print("\n" + "="*70)
    print("READING SELECTED DISTANCES")
    print("="*70)
    
    d_numbers = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            
            parts = line.split()
            if len(parts) >= 2:
                d_str = parts[1]  
                if d_str.startswith('d'):
                    d_num = int(d_str[1:]) 
                    d_numbers.append(d_num)
    
    print(f"Loaded {len(d_numbers):,} selected distances")
    print(f" Range: d{min(d_numbers)} to d{max(d_numbers)}")
    print(f" First 10: d{d_numbers[0]}, d{d_numbers[1]}, d{d_numbers[2]}, ...")
    
    return d_numbers


def parse_original_plumed(filepath):
    """
    Parse the original PLUMED file to extract all distance definitions.
    
    Returns:
        dict: Mapping from d-number to full PLUMED line
              {1: "d1: DISTANCE ATOMS=2070,2080  # PROA_ALA134 - PROA_ALA135", ...}
    """
    print("\n" + "="*70)
    print("PARSING ORIGINAL PLUMED FILE")
    print("="*70)
    
    distance_lines = {}
    other_lines = []  
    
    distance_pattern = re.compile(r'^d(\d+):\s*DISTANCE\s+ATOMS=(.+)$', re.IGNORECASE)
    
    with open(filepath, 'r') as f:
        for line in f:
            line_stripped = line.rstrip()
            match = distance_pattern.match(line_stripped)
            if match:
                d_num = int(match.group(1))
                distance_lines[d_num] = line_stripped
            else:
                if line_stripped and not line_stripped.startswith('#'):
                    other_lines.append(line_stripped)
    
    print(f"Found {len(distance_lines):,} distance definitions")
    print(f"Range: d{min(distance_lines.keys())} to d{max(distance_lines.keys())}")
    print(f"Found {len(other_lines)} other PLUMED commands")
    
    return distance_lines, other_lines


def create_new_plumed(selected_d_numbers, distance_lines, other_lines, output_path):
    """
    Create new PLUMED file with only selected distances in correct order.
    """
    print("\n" + "="*70)
    print("CREATING NEW PLUMED FILE")
    print("="*70)
    
    missing_distances = []
    selected_lines = []
    
    for idx, d_num in enumerate(selected_d_numbers):
        if d_num in distance_lines:
            selected_lines.append(distance_lines[d_num])
        else:
            missing_distances.append(d_num)
            print(f"Warning: d{d_num} not found in original file")
    
    if missing_distances:
        print(f"\nMissing {len(missing_distances)} distances!")
        print(f"  First missing: d{missing_distances[0]}")
        if len(missing_distances) > 10:
            print(f"  ... and {len(missing_distances)-1} more")
    
    with open(output_path, 'w') as f:
        f.write("# PLUMED input file for DeepTDA Collective Variable\n")
        f.write("# Generated automatically from feature selection\n")
        f.write(f"# Number of distances: {len(selected_lines):,}\n")
        f.write("#\n")
        f.write("# IMPORTANT: These distances MUST be fed to the PyTorch model\n")
        f.write("# in this exact order for correct CV calculation!\n")
        f.write("#\n\n")
        
        f.write("# Selected distance definitions (in required order)\n")
        for line in selected_lines:
            f.write(line + "\n")
        
        f.write("\n")
        
        f.write("# Collective variable calculation using PyTorch model\n")
        arg_list = ",".join([f"d{d_num}" for d_num in selected_d_numbers if d_num in distance_lines])
        
        f.write("cv: PYTORCH_MODEL ...\n")
        f.write("  FILE=deeptda_3state_final.ptc\n")
        f.write(f"  ARG={arg_list}\n")
        f.write("...\n\n")
        
        f.write("# Output the CV value\n")
        f.write("PRINT FILE=cv_output.dat STRIDE=12500 ARG=cv.node-0\n")
        f.write("\n")
        
        #for testing
        f.write("# Optional: Print all distances for verification\n")
        f.write(f"# PRINT FILE=distances.dat STRIDE=12500 ARG={arg_list}\n")
    
    print(f"Created new PLUMED file: {output_path}")
    print(f" Contains {len(selected_lines):,} distances")
    print(f" Ready for PyTorch CV calculation")


def create_summary(selected_d_numbers, distance_lines, missing, output_path):
    """Create a summary file with statistics and verification info."""
    print("\n" + "="*70)
    print("CREATING SUMMARY")
    print("="*70)
    
    with open(output_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("PLUMED DISTANCE EXTRACTION SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Total selected distances: {len(selected_d_numbers):,}\n")
        f.write(f"Successfully extracted: {len(selected_d_numbers) - len(missing):,}\n")
        f.write(f"Missing from original: {len(missing)}\n\n")
        
        if missing:
            f.write("Missing distances:\n")
            for d_num in missing[:20]: 
                f.write(f"  d{d_num}\n")
            if len(missing) > 20:
                f.write(f"  ... and {len(missing)-20} more\n")
            f.write("\n")
        
        f.write("Distance range:\n")
        valid_d = [d for d in selected_d_numbers if d in distance_lines]
        if valid_d:
            f.write(f"  Minimum: d{min(valid_d)}\n")
            f.write(f"  Maximum: d{max(valid_d)}\n")
            f.write(f"  Mean: d{int(np.mean(valid_d))}\n")
            f.write(f"  Median: d{int(np.median(valid_d))}\n\n")
        
        f.write("First 20 selected distances:\n")
        for i, d_num in enumerate(selected_d_numbers[:20]):
            status = "t" if d_num in distance_lines else "f"
            f.write(f"  Column {i:4d}: d{d_num:7d} {status}\n")
        
        if len(selected_d_numbers) > 20:
            f.write(f"  ... and {len(selected_d_numbers)-20} more\n")
    
    print(f"Summary saved: {output_path}")


def main():
    config = CONFIG
    
    print("="*70)
    print("EXTRACT SELECTED DISTANCES FOR PLUMED CV")
    print("="*70)
    
    if not Path(config['original_plumed']).exists():
        print(f"Error: Original PLUMED file not found: {config['original_plumed']}")
        return
    
    if not Path(config['selected_distances']).exists():
        print(f"rror: Selected distances file not found: {config['selected_distances']}")
        return
    
    selected_d_numbers = parse_selected_distances(config['selected_distances'])
    
    distance_lines, other_lines = parse_original_plumed(config['original_plumed'])
    
    missing = [d for d in selected_d_numbers if d not in distance_lines]
    if missing:
        print(f"\n WARNING: {len(missing)} selected distances not found in original file!")
    
    create_new_plumed(selected_d_numbers, distance_lines, other_lines, 
                     config['output_plumed'])
    
    if config['create_summary']:
        create_summary(selected_d_numbers, distance_lines, missing,
                      config['summary_file'])
    
    print("\n" + "="*70)
    print("EXTRACTION COMPLETE")
    print("="*70)
    print(f"Selected: {len(selected_d_numbers):,} distances")
    print(f"Extracted: {len(selected_d_numbers) - len(missing):,} distances")
    if missing:
        print(f"Missing: {len(missing)} distances")
    print(f"\nOutput file: {config['output_plumed']}")
    print("="*70)


if __name__ == "__main__":
    main()