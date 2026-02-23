"""
RAM-Aware Feature Selection for Large Distance Matrices

This script processes multiple .npy files (from MD simulations) and performs
feature selection without loading all data into memory at once.

Workflow: Compute statistics (variance, F-scores) in a streaming fashion,
Select top K features based on statistics,
Create reduced datasets by reading and writing in chunks
"""

import numpy as np
from pathlib import Path
from sklearn.feature_selection import f_classif
import pickle
import _pickle
from datetime import datetime
from tqdm import tqdm
import json

CONFIG = {
    # Input: List of (filepath, shape, label) tuples
    # shape: (n_samples, n_features) for each file
    # label: 0=Resting, 1=Open, 2=Desensitized
    'input_files': [
        ('example/Bicu/dist1.npy', (8000, 1405326), 0),
        ('example/Bicu/dist2.npy', (8000, 1405326), 0),
        ('example/Bicu/dist3.npy', (8000, 1405326), 0),
        ('example/GABA/dist1.npy', (8000, 1405326), 1),
        ('example/GABA/dist2.npy', (8000, 1405326), 1),
        ('example/GABA/dist3.npy', (8000, 1405326), 1),
        ('example/Picro/dist1.npy', (8000, 1405326), 2),
        ('example/Picro/dist2.npy', (8000, 1405326), 2),
        ('example/Picro/dist3.npy', (8000, 1405326), 2),
    ],
    
    'output_full_data': 'full_dataset_reduced.npz',
    'output_subset_data': 'consolidated_3state_data_reduced.npz',
    'selector_file': 'feature_selector.pkl',
    'feature_indices_file': 'selected_feature_indices.npy',
    'stats_file': 'feature_selection_stats.json',
    'selected_distances_file': 'selected_distances.txt',  # Simple list of d values
    
    # Feature selection
    'variance_threshold': 0.0,  # Remove features with variance below this
    'n_features_final': 10000,  # Number of features to keep
    
    # Memory management
    'chunk_size': 1000,  # Frames to process at once 
    'max_ram_gb': 8,  # Maximum RAM 
    
    # Subset for hyperparameter tuning
    'create_tuning_subset': True,
    'subset_samples_per_state': 800,  # 800 per state × 3 states = 2,400 total
    'subset_random_seed': 42,
}


def check_input_files(file_list):
    """Validate that all input files exist and get their shapes."""
    print("\n" + "="*70)
    print("CHECKING INPUT FILES")
    print("="*70)
    
    file_info = []
    total_samples = 0
    n_features = None
    
    for filepath, shape, label in file_list:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Load as raw memmap 
        n_samples, n_feat = shape
        arr = np.memmap(filepath, dtype=np.float32, mode='r', shape=shape)
        
        if n_features is None:
            n_features = n_feat
        elif n_features != n_feat:
            raise ValueError(f"Feature mismatch: {filepath} has {n_feat} features, expected {n_features}")
        
        file_info.append({
            'path': filepath,
            'shape': shape,
            'label': label,
            'n_samples': n_samples,
            'size_gb': path.stat().st_size / 1e9
        })
        total_samples += n_samples
        
        print(f" Got {path.name}")
        print(f"  Shape: ({n_samples:,}, {n_feat:,})")
        print(f"  Size: {file_info[-1]['size_gb']:.2f} GB")
        print(f"  Label: {label}")
    
    print(f"\n{'='*70}")
    print(f"Total samples: {total_samples:,}")
    print(f"Total features: {n_features:,} (d1 to d{n_features})")
    print(f"Total data size: {sum(f['size_gb'] for f in file_info):.2f} GB")
    print(f"{'='*70}")
    
    return file_info, total_samples, n_features


def compute_statistics_streaming(file_info, n_features, chunk_size=1000):
    """
    Compute mean, variance, and class-wise statistics in streaming fashion.
    Uses Welford's online algorithm for numerical stability.
    """
    print("\n" + "="*70)
    print("COMPUTING FEATURE STATISTICS (STREAMING)")
    print("="*70)
 
    # Overall statistics
    n_total = sum(f['n_samples'] for f in file_info)
    mean_overall = np.zeros(n_features, dtype=np.float64)
    m2_overall = np.zeros(n_features, dtype=np.float64)
    
    # Per-class statistics for F-test
    n_classes = 3
    class_counts = np.zeros(n_classes, dtype=np.int64)
    class_means = np.zeros((n_classes, n_features), dtype=np.float64)
    class_m2 = np.zeros((n_classes, n_features), dtype=np.float64)
    
    samples_processed = 0
    
    # First pass: compute means and variances
    print("\nPass 1: Computing means and variances...")
    for file_dict in tqdm(file_info, desc="Processing files"):
        # Load as raw memmap
        data = np.memmap(file_dict['path'], dtype=np.float32, mode='r', shape=file_dict['shape'])
        label = file_dict['label']
        n_samples = file_dict['n_samples']
        
        # Process in chunks
        for start_idx in range(0, n_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, n_samples)
            chunk = np.array(data[start_idx:end_idx], dtype=np.float32)
            chunk_size_actual = chunk.shape[0]
            
            # Update overall statistics (Welford's algorithm)
            for row in chunk:
                samples_processed += 1
                delta = row - mean_overall
                mean_overall += delta / samples_processed
                delta2 = row - mean_overall
                m2_overall += delta * delta2
            
            class_counts[label] += chunk_size_actual
            for row in chunk:
                delta = row - class_means[label]
                class_means[label] += delta / class_counts[label]
                delta2 = row - class_means[label]
                class_m2[label] += delta * delta2
    
    variance_overall = m2_overall / (samples_processed - 1)
    
    print(f"Processed {samples_processed:,} samples")
    print(f"Overall variance: min={variance_overall.min():.6f}, max={variance_overall.max():.6f}")
    
    # Compute F-statistics for ANOVA
    print("\nPass 2: Computing F-statistics (ANOVA)...")
    
    # Between-group variance
    grand_mean = mean_overall
    ss_between = np.zeros(n_features, dtype=np.float64)
    for c in range(n_classes):
        ss_between += class_counts[c] * (class_means[c] - grand_mean) ** 2
    ms_between = ss_between / (n_classes - 1)
    
    # Within-group variance
    ss_within = np.sum(class_m2, axis=0)
    ms_within = ss_within / (samples_processed - n_classes)
    
    # F-statistic
    f_scores = ms_between / (ms_within + 1e-10)  # Add to avoid division by zero
    
    print(f"F-statistics: min={f_scores.min():.2f}, max={f_scores.max():.2f}, mean={f_scores.mean():.2f}")
    
    return {
        'variance': variance_overall,
        'f_scores': f_scores,
        'mean': mean_overall,
        'class_means': class_means,
        'class_counts': class_counts,
        'n_total': samples_processed
    }


def select_features(stats, config):
    """Select features based on variance and F-scores."""
    print("\n" + "="*70)
    print("SELECTING FEATURES")
    print("="*70)
    
    n_features = len(stats['variance'])
    
    # Variance threshold
    if config['variance_threshold'] > 0:
        var_mask = stats['variance'] >= config['variance_threshold']
        n_passed_var = var_mask.sum()
        print(f"Variance threshold: {config['variance_threshold']}")
        print(f"  Passed: {n_passed_var:,} / {n_features:,} ({n_passed_var/n_features*100:.1f}%)")
    else:
        var_mask = np.ones(n_features, dtype=bool)
        n_passed_var = n_features
        print("No variance filtering (threshold=0)")
    
    # top K by F-score among those passing variance threshold
    f_scores_filtered = stats['f_scores'].copy()
    f_scores_filtered[~var_mask] = -np.inf  
    
    k = min(config['n_features_final'], n_passed_var)
    top_k_indices = np.argsort(f_scores_filtered)[-k:][::-1]  # Descending order
    
    print(f"\nTop {k} features by F-score:")
    print(f"  F-score range: [{stats['f_scores'][top_k_indices[-1]]:.2f}, {stats['f_scores'][top_k_indices[0]]:.2f}]")
    print(f"  Distance range: d{top_k_indices[0]+1} to d{top_k_indices[-1]+1}")
    
    selection_mask = np.zeros(n_features, dtype=bool)
    selection_mask[top_k_indices] = True
    
    return selection_mask, top_k_indices


def create_reduced_datasets(file_info, selected_indices, config):
    """
    Create reduced datasets by reading selected features only.
    """
    print("\n" + "="*70)
    print("CREATING REDUCED DATASETS")
    print("="*70)
    
    n_total = sum(f['n_samples'] for f in file_info)
    n_features_selected = len(selected_indices)
    
    print(f"Creating full dataset: {n_total:,} samples × {n_features_selected:,} features")
    
    full_data_path = Path(config['output_full_data']).with_suffix('.dat')
    full_data = np.memmap(full_data_path, dtype=np.float32, mode='w+', 
                          shape=(n_total, n_features_selected))
    full_labels = np.zeros(n_total, dtype=np.int64)
    
    chunk_size = config['chunk_size']
    global_idx = 0
    
    print("\nWriting reduced data...")
    for file_dict in tqdm(file_info, desc="Processing files"):
        data = np.memmap(file_dict['path'], dtype=np.float32, mode='r', shape=file_dict['shape'])
        label = file_dict['label']
        n_samples = file_dict['n_samples']
        
        for start_idx in range(0, n_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, n_samples)
            chunk = np.array(data[start_idx:end_idx], dtype=np.float32)
            chunk_reduced = chunk[:, selected_indices]
            
            chunk_size_actual = chunk_reduced.shape[0]
            full_data[global_idx:global_idx+chunk_size_actual] = chunk_reduced
            full_labels[global_idx:global_idx+chunk_size_actual] = label
            global_idx += chunk_size_actual
    
    # Flush to disk
    full_data.flush()
    del full_data  # Close memmap

    print(f"\nConverting to compressed NPZ format...")
    full_data_reload = np.memmap(full_data_path, dtype=np.float32, mode='r', 
                                  shape=(n_total, n_features_selected))
    
    # release memmap
    print("Loading reduced data into memory for compression...")
    full_data_array = np.array(full_data_reload, dtype=np.float32)
    del full_data_reload  
    
    np.savez_compressed(
        config['output_full_data'],
        data=full_data_array,
        labels=full_labels
    )
    
    print(f"Full dataset saved: {config['output_full_data']}")
    print(f" Shape: ({n_total}, {n_features_selected})")
    
    try:
        full_data_path.unlink()
        print(f"Cleaned up temporary file: {full_data_path}")
    except PermissionError:
        print(f"Could not delete temporary file: {full_data_path}")
    
    return full_data_array, full_labels


def create_subset(full_data, full_labels, config):
    """Create stratified subset for hyperparameter tuning."""
    print("\n" + "="*70)
    print("CREATING HYPERPARAMETER TUNING SUBSET")
    print("="*70)
    
    samples_per_state = config['subset_samples_per_state']
    rng = np.random.RandomState(config['subset_random_seed'])
    
    subset_indices = []
    for state in [0, 1, 2]:
        state_indices = np.where(full_labels == state)[0]
        if len(state_indices) < samples_per_state:
            print(f"Warning: State {state} has only {len(state_indices)} samples, using all")
            selected = state_indices
        else:
            selected = rng.choice(state_indices, size=samples_per_state, replace=False)
        subset_indices.extend(selected)
    
    subset_indices = np.array(subset_indices)
    rng.shuffle(subset_indices)
    
    print(f"Extracting {len(subset_indices)} samples...")
    subset_data = np.array(full_data[subset_indices], dtype=np.float32)
    subset_labels = full_labels[subset_indices]
    
    unique, counts = np.unique(subset_labels, return_counts=True)
    print("\nSubset class distribution:")
    for cls, count in zip(unique, counts):
        print(f"  State {cls}: {count:,} ({count/len(subset_labels)*100:.1f}%)")
    
    # Save
    np.savez_compressed(
        config['output_subset_data'],
        data=subset_data,
        labels=subset_labels
    )
    
    print(f"Subset saved: {config['output_subset_data']}")
    print(f"  Shape: {subset_data.shape}")
    print(f"  Size: {Path(config['output_subset_data']).stat().st_size / 1e6:.1f} MB")


def save_selected_distances(selected_indices, config):
    """
    Save the list of selected distance indices (d values).
    Simple text format: one d-number per line.
    """
    print("\n" + "="*70)
    print("SAVING SELECTED DISTANCE IDENTIFIERS")
    print("="*70)
    
    # Convert 0-based indices to 1-based d-numbers
    d_numbers = selected_indices + 1
    
    with open(config['selected_distances_file'], 'w') as f:
        f.write("# Selected distances for DeepTDA CV\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Total distances: {len(d_numbers)}\n")
        f.write("#\n")
        f.write("# Column in reduced dataset -> Original distance identifier\n")
        f.write("# Format: Column_Index  d_Number\n")
        f.write("#\n")
        
        for new_idx, d_num in enumerate(d_numbers):
            f.write(f"{new_idx:6d}  d{d_num}\n")
    
    print(f"Distance identifiers saved: {config['selected_distances_file']}")
    print(f"  Format: Column index → d-number")
    print(f"  Example: Column 0 → d{d_numbers[0]}")
    print(f"  Example: Column {len(d_numbers)-1} → d{d_numbers[-1]}")
    
    # For easy loading
    np.save(config['selected_distances_file'].replace('.txt', '_dnums.npy'), d_numbers)
    print(f"Also saved as: {config['selected_distances_file'].replace('.txt', '_dnums.npy')}")


def save_metadata(stats, selected_indices, config, file_info):
    """Save feature selection metadata."""
    print("\n" + "="*70)
    print("SAVING METADATA")
    print("="*70)
    
    # Save feature selector info
    selector_data = {
        'selected_indices': selected_indices,
        'n_features_original': len(stats['variance']),
        'n_features_selected': len(selected_indices),
        'variance_threshold': config['variance_threshold'],
        'f_scores': stats['f_scores'][selected_indices],
        'config': config,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(config['selector_file'], 'wb') as f:
        pickle.dump(selector_data, f)
    print(f"Feature selector saved: {config['selector_file']}")
    
    # Save selected indices
    np.save(config['feature_indices_file'], selected_indices)
    print(f"Feature indices saved: {config['feature_indices_file']}")
    
    # Save selected distances
    save_selected_distances(selected_indices, config)
    
    # Save statistics summary
    stats_summary = {
        'n_files': len(file_info),
        'n_samples_total': sum(f['n_samples'] for f in file_info),
        'n_features_original': len(stats['variance']),
        'n_features_selected': len(selected_indices),
        'selected_d_range': [int(selected_indices[0] + 1), int(selected_indices[-1] + 1)],
        'f_score_range': [float(stats['f_scores'][selected_indices].min()),
                         float(stats['f_scores'][selected_indices].max())],
        'f_score_mean': float(stats['f_scores'][selected_indices].mean()),
        'variance_range': [float(stats['variance'][selected_indices].min()),
                          float(stats['variance'][selected_indices].max())],
        'class_counts': {int(k): int(v) for k, v in enumerate(stats['class_counts'])},
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(config['stats_file'], 'w') as f:
        json.dump(stats_summary, f, indent=2)
    print(f"Statistics saved: {config['stats_file']}")


def main():
    config = CONFIG
    
    print("="*70)
    print("RAM-AWARE FEATURE SELECTION FOR MD DISTANCE DATA")
    print("="*70)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Max RAM usage: ~{config['max_ram_gb']} GB")
    print(f"Target features: {config['n_features_final']:,}")
    
    # Check inputs
    file_info, n_total, n_features = check_input_files(config['input_files'])
    
    # Estimate memory usage, care float 32
    bytes_per_chunk = config['chunk_size'] * n_features * 4  
    ram_per_chunk_gb = bytes_per_chunk / 1e9
    print(f"\nEstimated RAM per chunk: {ram_per_chunk_gb:.2f} GB")
    if ram_per_chunk_gb > config['max_ram_gb']:
        suggested_chunk = int(config['chunk_size'] * config['max_ram_gb'] / ram_per_chunk_gb)
        print(f"⚠ WARNING: Chunk size too large! Reduce to ~{suggested_chunk}")
        return
    
    # Compute statistics
    stats = compute_statistics_streaming(file_info, n_features, config['chunk_size'])
    
    # Select features
    selection_mask, selected_indices = select_features(stats, config)
    
    # Create reduced datasets
    full_data, full_labels = create_reduced_datasets(file_info, selected_indices, config)
    
    # Create subset for hyperparameter tuning
    if config['create_tuning_subset']:
        create_subset(full_data, full_labels, config)
    
    # Save metadata
    save_metadata(stats, selected_indices, config, file_info)
    
    # Final 
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Features: {n_features:,} → {len(selected_indices):,} ({n_features/len(selected_indices):.1f}x reduction)")
    print(f"Selected: d{selected_indices[0]+1} to d{selected_indices[-1]+1}")
    print(f"Full dataset: {n_total:,} samples")
    if config['create_tuning_subset']:
        print(f"Tuning subset: {config['subset_samples_per_state']*3:,} samples")
    print(f"\nOutputs:")
    print(f"{config['output_full_data']}")
    if config['create_tuning_subset']:
        print(f"{config['output_subset_data']}")
    print(f"{config['selector_file']}")
    print(f"{config['feature_indices_file']}")
    print(f"{config['selected_distances_file']} ← Distance identifiers (d-numbers)")
    
 


if __name__ == "__main__":
    main()