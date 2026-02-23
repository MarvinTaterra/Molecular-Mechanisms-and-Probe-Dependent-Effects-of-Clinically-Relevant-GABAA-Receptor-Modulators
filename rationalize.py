"""
Feature importance analysis for trained DeepTDA model.
Identifies which C-alpha distances are most important for state discrimination.
Memory-efficient version with batch processing.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime

from mlcolvar.cvs import DeepTDA

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300


CONFIG = {
    'model_checkpoint': 'path/to/deeptda_3state_final_best.ckpt',
    'data_file': 'path/to/full_dataset_reduced.npz',
    'selected_distances_file': 'path/to/selected_distances.txt',
    'output_dir': 'path/to/feature_importance_analysis',
    'top_n_features': 100,
    'n_integration_steps': 30,  
    'max_samples': 2000,  
    'gradient_batch_size': 50,  
    'ig_batch_size': 25, 
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}



def load_distance_mapping(selected_distances_file):
    """
    Load the mapping from column index to d_number.
    
    Returns:
    --------
    mapping : dict
        {column_index: 'd_number'}
    """
    mapping = {}
    
    with open(selected_distances_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            
            parts = line.split()
            if len(parts) == 2:
                col_idx = int(parts[0])
                d_number = parts[1]
                mapping[col_idx] = d_number
    
    print(f"Loaded {len(mapping)} distance mappings")
    return mapping


def compute_feature_importance_gradients(model, data, device='cpu', batch_size=100):
    """
    Compute feature importance using input gradients (memory-efficient).
    
    Parameters:
    -----------
    model : DeepTDA model
    data : np.array
        Input data [n_samples, n_features]
    batch_size : int
        Process this many samples at a time
    
    Returns:
    --------
    importance : np.array [n_features]
        Mean absolute gradient for each feature
    """
    model.eval()
    
    # Convert to tensor
    if isinstance(data, np.ndarray):
        data_tensor = torch.tensor(data, dtype=torch.float32)
    else:
        data_tensor = data
    
    n_samples = data_tensor.shape[0]
    n_features = data_tensor.shape[1]
    
    print(f"Computing gradients in batches of {batch_size}...")
    
    # Accumulator for gradients
    gradient_sum = np.zeros(n_features, dtype=np.float64)
    n_processed = 0
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_data = data_tensor[start_idx:end_idx].to(device)
        batch_data.requires_grad = True
        
        if start_idx % 500 == 0:
            print(f"  Processing samples {start_idx}-{end_idx}/{n_samples}")
        
        output = model(batch_data)
        
        for i in range(output.shape[0]):
            model.zero_grad()
            if batch_data.grad is not None:
                batch_data.grad.zero_()
            
            output[i].backward(retain_graph=(i < output.shape[0] - 1))
            
            gradient_sum += batch_data.grad[i].abs().cpu().numpy()
            n_processed += 1

        del batch_data, output
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    importance = gradient_sum / n_processed
    
    return importance


def integrated_gradients(model, data, baseline=None, steps=50, device='cpu', batch_size=50):
    """
    Compute integrated gradients for feature importance (memory-efficient).
    
    Parameters:
    -----------
    model : DeepTDA model
    data : torch.Tensor [n_samples, n_features]
    baseline : torch.Tensor or None
        Baseline input (default: zeros)
    steps : int
        Number of integration steps
    batch_size : int
        Process this many samples at a time
    
    Returns:
    --------
    attributions : np.array [n_samples, n_features]
    """
    model.eval()
    
    if isinstance(data, np.ndarray):
        data_tensor = torch.tensor(data, dtype=torch.float32)
    else:
        data_tensor = data
    
    n_samples = data_tensor.shape[0]
    n_features = data_tensor.shape[1]
    
    if baseline is None:
        baseline = torch.zeros_like(data_tensor)
    
    print(f"Computing integrated gradients with {steps} steps, batch size {batch_size}...")
    
    all_attributions = np.zeros((n_samples, n_features), dtype=np.float32)
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_data = data_tensor[start_idx:end_idx]
        batch_baseline = baseline[start_idx:end_idx]
        
        print(f"  Processing samples {start_idx}-{end_idx}/{n_samples}")
        alphas = torch.linspace(0, 1, steps)
        
        batch_gradient_sum = np.zeros((end_idx - start_idx, n_features), dtype=np.float64)
        
        for step_idx, alpha in enumerate(alphas):
            if step_idx % 10 == 0:
                print(f"    Alpha step {step_idx}/{steps}")
            
            interpolated = batch_baseline + alpha * (batch_data - batch_baseline)
            interpolated = interpolated.to(device)
            interpolated.requires_grad = True
            
            outputs = model(interpolated)
            
            for i in range(outputs.shape[0]):
                model.zero_grad()
                if interpolated.grad is not None:
                    interpolated.grad.zero_()
                
                outputs[i].backward(retain_graph=(i < outputs.shape[0] - 1))
                batch_gradient_sum[i] += interpolated.grad[i].cpu().numpy()
            
            del interpolated, outputs
            if device == 'cuda':
                torch.cuda.empty_cache()
        
        avg_gradients = batch_gradient_sum / steps
        
        input_diff = (batch_data - batch_baseline).numpy()
        batch_attributions = input_diff * avg_gradients
        
        all_attributions[start_idx:end_idx] = batch_attributions

        del batch_data, batch_baseline, batch_gradient_sum, avg_gradients
    
    return all_attributions


def analyze_feature_importance(model_checkpoint, data_file, selected_distances_file,
                               top_n=100, n_steps=50, device='cpu', 
                               max_samples=2000, gradient_batch_size=50, ig_batch_size=25):
    """
    Comprehensive feature importance analysis (memory-efficient).
    
    Parameters:
    -----------
    max_samples : int
        Maximum number of samples to use (reduces memory usage)
    gradient_batch_size : int
        Batch size for gradient computation
    ig_batch_size : int
        Batch size for integrated gradients (should be smaller)
    
    Returns:
    --------
    results : dict
        Dictionary containing feature importance analysis results
    """
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*70)
    
    print(f"\nLoading model from: {model_checkpoint}")
    model = DeepTDA.load_from_checkpoint(model_checkpoint)
    model = model.to(device)
    model.eval()
    print("Model loaded")
    
    print(f"\nLoading data from: {data_file}")
    with np.load(data_file) as npz:
        data = npz['data'].astype(np.float32)
        labels = npz['labels'].astype(np.int64)
    print(f"Data loaded: {data.shape}")
    
    distance_mapping = load_distance_mapping(selected_distances_file)
    
    n_samples_for_analysis = min(max_samples, len(data))
    print(f"\nUsing {n_samples_for_analysis} samples for analysis (to save memory)")
    
    sample_indices = []
    samples_per_class = n_samples_for_analysis // len(np.unique(labels))
    
    for label_val in np.unique(labels):
        class_indices = np.where(labels == label_val)[0]
        selected = np.random.choice(class_indices, 
                                   min(samples_per_class, len(class_indices)), 
                                   replace=False)
        sample_indices.extend(selected)
    
    sample_indices = np.array(sample_indices)
    data_sample = data[sample_indices]
    
    print(f"Actual samples used: {len(sample_indices)}")
    
    # Gradient-based 
    print("\n" + "-"*70)
    print(" Gradient-based importance")
    print("-"*70)
    grad_importance = compute_feature_importance_gradients(
        model, data_sample, device, batch_size=gradient_batch_size
    )
    print("Gradient-based importance computed")
    
    # Integrated gradients
    print("\n" + "-"*70)
    print("Integrated gradients")
    print("-"*70)
    ig_attributions = integrated_gradients(
        model, data_sample, steps=n_steps, device=device, batch_size=ig_batch_size
    )
    ig_importance = np.mean(np.abs(ig_attributions), axis=0)
    print("Integrated gradients computed")
    
    combined_importance = (grad_importance + ig_importance) / 2
    
    top_indices = np.argsort(combined_importance)[-top_n:][::-1]
    feature_names = []
    for idx in top_indices:
        if idx in distance_mapping:
            feature_names.append(distance_mapping[idx])
        else:
            feature_names.append(f"feature_{idx}")
    
    results = {
        'feature_indices': top_indices,
        'feature_names': feature_names,
        'distance_ids': [distance_mapping.get(idx, f"unknown_{idx}") for idx in top_indices],
        'importance_scores': combined_importance[top_indices],
        'gradient_scores': grad_importance[top_indices],
        'ig_scores': ig_importance[top_indices],
        'all_importance': combined_importance,
        'all_gradient': grad_importance,
        'all_ig': ig_importance
    }
    
    return results


def visualize_top_features(results, output_dir, top_k=20):
    """
    Visualize top important features.
    """
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    y_pos = np.arange(top_k)
    scores = results['importance_scores'][:top_k]
    names = results['distance_ids'][:top_k]
    
    bars = ax.barh(y_pos, scores, color='steelblue', alpha=0.8, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_k} Most Important C$\\alpha$-C$\\alpha$ Distances', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    
    plt.tight_layout()
    plot_file = output_dir / 'top_features_importance.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {plot_file}")
    plt.close()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    grad_scores = results['gradient_scores'][:top_k]
    ig_scores = results['ig_scores'][:top_k]
    
    x = np.arange(top_k)
    width = 0.35
    
    ax.barh(x - width/2, grad_scores, width, label='Gradient', alpha=0.8, color='coral')
    ax.barh(x + width/2, ig_scores, width, label='Integrated Gradient', alpha=0.8, color='steelblue')
    
    ax.set_yticks(x)
    ax.set_yticklabels([f"{i+1}. {name}" for i, name in enumerate(names)], fontsize=8)
    ax.set_xlabel('Importance Score', fontsize=11, fontweight='bold')
    ax.set_title('Method Comparison: Gradient vs Integrated Gradients', 
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    
    plt.tight_layout()
    plot_file = output_dir / 'method_comparison.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {plot_file}")
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))

    top_score = results['importance_scores'][0]
    ax.hist(results['all_importance'], bins=100, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(top_score, color='red', linestyle='--', 
               linewidth=2, label=f'Top feature: {top_score:.4f}')
    ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Feature Importance Scores', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plot_file = output_dir / 'importance_distribution.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {plot_file}")
    plt.close()


def save_results(results, output_dir):
    """Save results to files."""
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    npz_file = output_dir / 'feature_importance.npz'
    np.savez(npz_file,
             feature_indices=results['feature_indices'],
             importance_scores=results['importance_scores'],
             gradient_scores=results['gradient_scores'],
             ig_scores=results['ig_scores'],
             all_importance=results['all_importance'])
    print(f"Saved: {npz_file}")
    
    txt_file = output_dir / 'top_features.txt'
    with open(txt_file, 'w') as f:
        f.write("# Top Important Features for DeepTDA Model\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("#\n")
        f.write("# Rank  Distance_ID  Combined_Score  Gradient_Score  IG_Score\n")
        f.write("#" + "="*70 + "\n")
        
        for i, (dist_id, comb_score, grad_score, ig_score) in enumerate(
            zip(results['distance_ids'], results['importance_scores'], 
                results['gradient_scores'], results['ig_scores']), 1):
            f.write(f"{i:4d}  {dist_id:>10s}  {comb_score:14.6f}  {grad_score:14.6f}  {ig_score:14.6f}\n")
    
    print(f"Saved: {txt_file}")
    
    json_file = output_dir / 'feature_importance.json'
    json_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'n_features_analyzed': len(results['all_importance']),
        'top_features': [
            {
                'rank': i+1,
                'distance_id': dist_id,
                'feature_index': int(idx),
                'combined_score': float(comb_score),
                'gradient_score': float(grad_score),
                'ig_score': float(ig_score)
            }
            for i, (idx, dist_id, comb_score, grad_score, ig_score) in enumerate(
                zip(results['feature_indices'], results['distance_ids'], 
                    results['importance_scores'], results['gradient_scores'], 
                    results['ig_scores']))
        ]
    }
    
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved: {json_file}")


def print_summary(results, top_n=10):
    """Print summary of top features."""
    print("\n" + "="*70)
    print(f"TOP {top_n} MOST IMPORTANT FEATURES")
    print("="*70)
    print(f"{'Rank':<6} {'Distance ID':<12} {'Combined':<12} {'Gradient':<12} {'Int.Grad':<12}")
    print("-"*70)
    
    for i in range(min(top_n, len(results['distance_ids']))):
        print(f"{i+1:<6} {results['distance_ids'][i]:<12} "
              f"{results['importance_scores'][i]:<12.6f} "
              f"{results['gradient_scores'][i]:<12.6f} "
              f"{results['ig_scores'][i]:<12.6f}")
    
    print("="*70)


def main():
    config = CONFIG
    
    print("="*70)
    print("DeepTDA FEATURE IMPORTANCE ANALYSIS")
    print("="*70)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {config['device']}")
    
    results = analyze_feature_importance(
        model_checkpoint=config['model_checkpoint'],
        data_file=config['data_file'],
        selected_distances_file=config['selected_distances_file'],
        top_n=config['top_n_features'],
        n_steps=config['n_integration_steps'],
        device=config['device'],
        max_samples=config['max_samples'],
        gradient_batch_size=config['gradient_batch_size'],
        ig_batch_size=config['ig_batch_size']
    )
    
    print_summary(results, top_n=20)
    
    visualize_top_features(results, config['output_dir'], top_k=20)
    
    save_results(results, config['output_dir'])
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()