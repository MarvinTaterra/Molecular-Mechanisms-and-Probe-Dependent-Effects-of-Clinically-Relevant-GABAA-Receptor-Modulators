"""
Train final DeepTDA model on full dataset using best hyperparameters from Optuna.
Creates plots and exports model for PLUMED enhanced sampling.
"""

import torch
import lightning
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime

from mlcolvar.cvs import DeepTDA
from mlcolvar.data import DictModule, DictDataset
from mlcolvar.utils.trainer import MetricsCallback
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

# Set style for plot
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

CONFIG = {
    # Data
    'data_file': 'full_dataset_reduced.npz',
    'selected_distances_file': 'selected_distances.txt',
    
    # Best hyperparameters from Optuna
    'best_config_file': 'FOLDER/TRIALNAME_best.json',
    
    # Training
    'batch_size': 2048,  
    'max_epochs': 500,
    'early_stopping_patience': 30,
    'early_stopping_min_delta': 0.0001,
    'validation_split': 0.15, 
    
    #Export
    'output_dir': 'final_model',
    'model_name': 'deeptda_3state_final',

    'seed': 42,
}



def load_best_hyperparameters(config_file):
    """Load best hyperparameters from Optuna optimization."""
    print("\n" + "="*70)
    print("LOADING BEST HYPERPARAMETERS")
    print("="*70)
    
    with open(config_file, 'r') as f:
        best_config = json.load(f)
    
    print(f"Loaded from: {config_file}")
    print(f"Best loss: {best_config['loss']:.6f}")
    print(f"Architecture: {best_config['architecture']}")
    print(f"Learning rate: {best_config['params']['learning_rate']:.6f}")
    print(f"Activation: {best_config['params']['activation']}")
    print(f"Weight decay: {best_config['params']['weight_decay']:.6f}")
    
    return best_config


def load_data(data_file):
    """Load the full reduced dataset."""
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    print(f"File: {data_file}")
    
    with np.load(data_file) as npz:
        data = npz['data'].astype(np.float32)
        labels = npz['labels'].astype(np.int64)
    
    print(f"Shape: {data.shape}")
    print(f"Memory: {data.nbytes / 1e9:.2f} GB")
    
    unique, counts = np.unique(labels, return_counts=True)
    print("\nClass distribution:")
    for cls, count in zip(unique, counts):
        state_name = ['Resting', 'Open', 'Desensitized'][cls]
        print(f"  {state_name} ({cls}): {count:,} ({count/len(labels)*100:.1f}%)")
    
    return data, labels


def create_model(n_features, best_config):
    """Create DeepTDA model with best hyperparameters."""
    print("\n" + "="*70)
    print("CREATING MODEL")
    print("="*70)
    
    # Extract parameters
    params = best_config['params']
    architecture = best_config['architecture']
    n_states = best_config['n_states']
    n_cvs = best_config['n_cvs']
    target_centers = best_config['target_centers']
    target_sigmas = best_config['target_sigmas']
    
    # Build layer specification: [input, hidden..., output]
    layers = [n_features] + architecture + [n_cvs]
    
    # Create model WITHOUT input normalization to avoid the dtype issue
    model = DeepTDA(
        n_states=n_states,
        n_cvs=n_cvs,
        target_centers=target_centers,
        target_sigmas=target_sigmas,
        layers=layers,
        options={
            'nn': {'activation': params['activation']},
            'norm_in': None  # Disable input normalization
        }
    )
    
    # Set optimizer parameters
    model.optimizer_name = 'Adam'
    model.optimizer_kwargs = {
        'lr': params['learning_rate'],
        'weight_decay': params['weight_decay']
    }
    
    print(f"Architecture: {layers}")
    print(f"Activation: {params['activation']}")
    print(f"Target centers: {target_centers}")
    print(f"Target sigmas: {target_sigmas}")
    print(f"Learning rate: {params['learning_rate']:.6f}")
    print(f"Weight decay: {params['weight_decay']:.6f}")
    print(f"Input normalization: Disabled")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    return model


def train_model(model, data, labels, config):
    """Train the model using Lightning."""
    print("\n" + "="*70)
    print("TRAINING MODEL")
    print("="*70)
    
    # ensure data is float and labels are long
    dataset = DictDataset({
        'data': torch.FloatTensor(data),
        'labels': torch.LongTensor(labels)
    })
    
    # Create datamodule with train/val split
    datamodule = DictModule(
        dataset,
        lengths=[1.0 - config['validation_split'], config['validation_split']],
        batch_size=config['batch_size'],
        shuffle=True
    )
    
    print(f"Batch size: {config['batch_size']}")
    print(f"Train samples: {int(len(data) * (1 - config['validation_split'])):,}")
    print(f"Val samples: {int(len(data) * config['validation_split']):,}")
    print(f"Max epochs: {config['max_epochs']}")
    
    # Setup callbacks
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(exist_ok=True, parents=True)
    
    metrics = MetricsCallback()
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename=f"{config['model_name']}_best",
        monitor='valid_loss',
        mode='min',
        save_top_k=1,
        save_last=True
    )
    
    early_stopping = EarlyStopping(
        monitor='valid_loss',
        patience=config['early_stopping_patience'],
        min_delta=config['early_stopping_min_delta'],
        mode='min',
        verbose=True
    )
    
    # Create trainer
    trainer = lightning.Trainer(
        max_epochs=config['max_epochs'],
        callbacks=[metrics, checkpoint_callback, early_stopping],
        accelerator='auto',
        devices=1,
        logger=False,
        enable_progress_bar=True,
        gradient_clip_val=1.0  # Prevent exploding gradients
    )
    
    # Train
    print("\nStarting training...")
    trainer.fit(model, datamodule)
    
    print(f"\nTraining complete!")
    print(f"Best validation loss: {checkpoint_callback.best_model_score:.6f}")
    print(f"Best model saved: {checkpoint_callback.best_model_path}")
    
    return model, metrics, checkpoint_callback


def plot_training_metrics(metrics, output_dir):
    """Plot training and validation loss curves."""
    print("\n" + "="*70)
    print("CREATING TRAINING PLOTS")
    print("="*70)
    
    output_dir = Path(output_dir)
    
    # Extract metrics
    train_loss = metrics.metrics.get('train_loss', [])
    valid_loss = metrics.metrics.get('valid_loss', [])
    
    if not train_loss or not valid_loss:
        print("No metrics to plot")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(train_loss) + 1)
    ax.plot(epochs, train_loss, label='Training Loss', linewidth=2, alpha=0.8)
    ax.plot(epochs, valid_loss, label='Validation Loss', linewidth=2, alpha=0.8)
    
    # Mark best epoch
    best_epoch = np.argmin(valid_loss) + 1
    best_loss = min(valid_loss)
    ax.axvline(best_epoch, color='red', linestyle='--', alpha=0.5, 
               label=f'Best Epoch ({best_epoch})')
    ax.scatter(best_epoch, best_loss, color='red', s=100, zorder=5)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('DeepTDA Training Progress', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Log scale if range is large
    if max(train_loss) / min(train_loss) > 10:
        ax.set_yscale('log')
    
    plt.tight_layout()
    plot_path = output_dir / 'training_loss.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Training plot saved: {plot_path}")
    plt.close()


def analyze_predictions(model, data, labels, output_dir):
    """Analyze model predictions and create visualizations."""
    print("\n" + "="*70)
    print("ANALYZING MODEL PREDICTIONS")
    print("="*70)
    
    output_dir = Path(output_dir)
    
    # Move model to CPU for inference to avoid device issues
    model = model.cpu()
    model.eval()
    
    # Get predictions in batches to avoid memory issues
    batch_size = 2048
    cv_values = []
    
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = torch.FloatTensor(data[i:i+batch_size])
            batch_cv = model(batch).cpu().numpy().flatten()
            cv_values.append(batch_cv)
    
    cv_values = np.concatenate(cv_values)
    
    # Statistics by state
    print("\nCV statistics by state:")
    state_names = ['Resting', 'Open', 'Desensitized']
    target_centers = [-4.0, 0.0, 4.0]
    
    for state_idx in range(3):
        state_mask = labels == state_idx
        state_cv = cv_values[state_mask]
        
        print(f"\n{state_names[state_idx]} (target: {target_centers[state_idx]}):")
        print(f"  Mean: {state_cv.mean():.3f}")
        print(f"  Std:  {state_cv.std():.3f}")
        print(f"  Min:  {state_cv.min():.3f}")
        print(f"  Max:  {state_cv.max():.3f}")
    
    # Create distribution plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    ax = axes[0]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    for state_idx in range(3):
        state_mask = labels == state_idx
        state_cv = cv_values[state_mask]
        
        ax.hist(state_cv, bins=100, alpha=0.6, label=state_names[state_idx],
                color=colors[state_idx], density=True)
        
        # Mark target center
        ax.axvline(target_centers[state_idx], color=colors[state_idx],
                   linestyle='--', linewidth=2, alpha=0.8,
                   label=f'{state_names[state_idx]} target')
    
    ax.set_xlabel('CV Value', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('CV Distribution by State', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    cv_by_state = [cv_values[labels == i] for i in range(3)]
    
    parts = ax.violinplot(cv_by_state, positions=[0, 1, 2], showmeans=True,
                          showmedians=True, widths=0.7)
    
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
    
    # Mark targets
    for i, (center, color) in enumerate(zip(target_centers, colors)):
        ax.scatter(i, center, color=color, s=200, marker='*',
                   edgecolors='black', linewidths=1.5, zorder=10,
                   label=f'Target: {center}')
    
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(state_names)
    ax.set_ylabel('CV Value', fontsize=12)
    ax.set_title('CV Distribution Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_path = output_dir / 'cv_distributions.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nDistribution plot saved: {plot_path}")
    plt.close()
    
    # Separation quality analysis
    print("\nSeparation quality:")
    for i in range(3):
        for j in range(i+1, 3):
            state_i_cv = cv_values[labels == i]
            state_j_cv = cv_values[labels == j]
            
            # Cohen's d (effect size)
            pooled_std = np.sqrt((state_i_cv.std()**2 + state_j_cv.std()**2) / 2)
            cohens_d = abs(state_i_cv.mean() - state_j_cv.mean()) / pooled_std
            
            print(f"  {state_names[i]} vs {state_names[j]}: Cohen's d = {cohens_d:.2f}")


def export_for_plumed(model, config, selected_distances_file, n_features):
    """Export model for PLUMED using TorchScript."""
    print("\n" + "="*70)
    print("EXPORTING MODEL FOR PLUMED")
    print("="*70)
    
    output_dir = Path(config['output_dir'])
    
    # Move model to CPU and set to evaluation mode
    model = model.cpu()
    model.eval()
    
    # Extract the actual PyTorch neural network from the FeedForward wrapper
    # The FeedForward object has a 'nn' attribute which is the actual Sequential
    class PurePyTorchWrapper(torch.nn.Module):
        def __init__(self, lightning_model):
            super().__init__()
            # Extract the actual Sequential from FeedForward.nn.nn
            if hasattr(lightning_model.nn, 'nn'):
                # FeedForward has a 'nn' attribute containing the Sequential
                self.network = lightning_model.nn.nn
            else:
                # Fallback
                self.network = lightning_model.nn
            
        def forward(self, x):
            return self.network(x)
    
    # Create wrapper
    pure_model = PurePyTorchWrapper(model)
    pure_model.eval()
    
    # Create example input
    example_input = torch.randn(1, n_features)
    
    # Wrapper testing
    print("Testing model wrapper...")
    with torch.no_grad():
        try:
            test_output = pure_model(example_input)
            print(f"Model wrapper test: output shape = {test_output.shape}")
        except Exception as e:
            print(f"Model wrapper test failed: {e}")
            checkpoint_file = config['model_name'] + '_best.ckpt'
            print(f"Please use checkpoint file: {output_dir / checkpoint_file}")
            return
    
    # Trace the pure PyTorch wrapper
    print("Tracing model with TorchScript...")
    try:
        traced_model = torch.jit.trace(pure_model, example_input)
        print("TorchScript tracing successful")
    except Exception as e:
        print(f"TorchScript tracing failed: {e}")
        print("Attempting torch.jit.script instead...")
        try:
            traced_model = torch.jit.script(pure_model)
            print("TorchScript scripting successful")
        except Exception as e2:
            print(f"TorchScript scripting also failed: {e2}")
            
            # try to trace just the Sequential directly
            print("Attempting to trace Sequential directly...")
            try:
                if hasattr(model.nn, 'nn'):
                    sequential_model = model.nn.nn
                else:
                    sequential_model = model.nn
                traced_model = torch.jit.trace(sequential_model, example_input)
                print("Direct Sequential tracing successful")
            except Exception as e3:
                print(f"All export methods failed: {e3}")
                checkpoint_file = config['model_name'] + '_best.ckpt'
                print(f"Using checkpoint file directly: {output_dir / checkpoint_file}")
                return
    
    # Save traced model
    model_path = output_dir / f"{config['model_name']}.ptc"
    torch.jit.save(traced_model, str(model_path))
    
    print(f"Model exported: {model_path}")
    print(f"Format: TorchScript (.ptc)")
    
    # Verify the exported model works
    print("Verifying exported model...")
    with torch.no_grad():
        original_output = pure_model(example_input)
        loaded_model = torch.jit.load(str(model_path))
        loaded_output = loaded_model(example_input)
        max_diff = torch.max(torch.abs(original_output - loaded_output)).item()
        print(f"Export verification: max difference = {max_diff:.2e}")
        if max_diff < 1e-6:
            print("Model export verified successfully!")
        else:
            print(f"Warning: Non-negligible difference detected ({max_diff:.2e})")
    
    # Save model info
    info_file = output_dir / f"{config['model_name']}_info.json"
    with open(info_file, 'w') as f:
        json.dump({
            'model_file': str(model_path.name),
            'n_inputs': n_features,
            'n_outputs': 1,
            'n_states': 3,
            'target_centers': [-4.0, 0.0, 4.0],
            'target_sigmas': [0.5, 0.5, 0.5],
            'selected_distances_file': selected_distances_file,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }, f, indent=2)
    
    print(f"Model info saved: {info_file}")
    
    
    # Save traced model
    model_path = output_dir / f"{config['model_name']}.ptc"
    torch.jit.save(traced_model, str(model_path))
    
    print(f"Model exported: {model_path}")
    print(f"Format: TorchScript (.ptc)")
    
    # Verify the exported model works
    print("Verifying exported model...")
    with torch.no_grad():
        original_output = pure_model(example_input)
        loaded_model = torch.jit.load(str(model_path))
        loaded_output = loaded_model(example_input)
        max_diff = torch.max(torch.abs(original_output - loaded_output)).item()
        print(f"Export verification: max difference = {max_diff:.2e}")
        if max_diff < 1e-6:
            print("Model export verified successfully!")
        else:
            print(f"Warning: Non-negligible difference detected ({max_diff:.2e})")
    
    # Save model info
    info_file = output_dir / f"{config['model_name']}_info.json"
    with open(info_file, 'w') as f:
        json.dump({
            'model_file': str(model_path.name),
            'n_inputs': n_features,
            'n_outputs': 1,
            'n_states': 3,
            'target_centers': [-4.0, 0.0, 4.0],
            'target_sigmas': [0.5, 0.5, 0.5],
            'selected_distances_file': selected_distances_file,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }, f, indent=2)
    
    print(f"Model info saved: {info_file}")


def main():
    config = CONFIG
    
    print("="*70)
    print("FINAL DeepTDA MODEL TRAINING")
    print("="*70)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Set seed
    lightning.seed_everything(config['seed'])
    
    # Load best hyperparameters
    best_config = load_best_hyperparameters(config['best_config_file'])
    
    # Load data
    data, labels = load_data(config['data_file'])
    n_features = data.shape[1]
    
    # Create model
    model = create_model(n_features, best_config)
    
    # Train model
    model, metrics, checkpoint = train_model(model, data, labels, config)
    
    # Load best model
    print("\nLoading best model checkpoint...")
    best_model = DeepTDA.load_from_checkpoint(checkpoint.best_model_path)
    
    # Plot training metrics
    plot_training_metrics(metrics, config['output_dir'])
    
    # Analyze predictions
    analyze_predictions(best_model, data, labels, config['output_dir'])
    
    # Export for PLUMED
    export_for_plumed(best_model, config, config['selected_distances_file'], n_features)
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Output directory: {config['output_dir']}/")
    print(f"Model: {config['model_name']}.ptc")
    print(f"Model info: {config['model_name']}_info.json")
    print(f"Plots: training_loss.png, cv_distributions.png")
    print("="*70)


if __name__ == "__main__":
    main()