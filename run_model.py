"""
Hyperparameter optimization for 3-state DeepTDA using Optuna with 5-fold cross-validation.

Features:
- Optimized for FULL dataset (72K samples * 10K features)
- Automatic batch size adjustment based on GPU memory
- Works on ANY GPU 
- Efficient memory management
- Stratified K-fold CV for 3 states
- Trains 1 CV to discriminate 3 states
- Aggressive pruning for faster exploration

"""

import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
import numpy as np
import optuna
from optuna.trial import TrialState
import json
from pathlib import Path
import time
from datetime import datetime
import gc
import sys
import math
import pickle

#Just for env lols
try:
    from model import create_deep_tda_model
    from mlcolvar.core.loss import TDALoss
except ImportError as e:
    print(f"ERROR: Could not import required modules: {e}")
    print("Make sure you have 'model.py' and 'mlcolvar' installed")
    sys.exit(1)


OPTIMIZATION_CONFIG = {
    'data_file': 'full_dataset_reduced.npz', 
    'n_trials': 25,  
    'n_folds': 5,    
    'n_epochs_per_fold': 100,  
    'timeout_hours': 24,
    'output_dir': 'optuna_results_3state_full',
    'study_name': None,
    'device': 'cuda',
    'batch_size': 512,  # auto-adjusted for GPU
    'enable_pruning': True,
    'pruning_warmup_steps': 3, 
    'pruning_check_interval': 10,  
    'random_seed': 42,
    'clear_cache_every_n_epochs': 5, 
    'print_progress_every_n_epochs': 25,  

    # Checkpointing settings
    'enable_checkpointing': True,
    'checkpoint_file': 'optuna_checkpoint_full.pkl',
    'checkpoint_freq_trials': 3, 
    'resume_from_checkpoint': True,
    # TDA-specific settings
    'n_cvs': 1, 
    'n_states': 3,  
    'target_centers': [-4.0, 0.0, 4],  # Target positions for 3 states on the CV
    'target_sigmas': [0.5, 0.5, 0.5],  # Target widths for 3 states
    'early_stopping_patience': 15,  
    'early_stopping_min_delta': 0.001, 
}

HYPERPARAMETER_SPACE = {
    'n_layers': {'type': 'int', 'low': 2, 'high': 4},
    'layer_size_min': 128,  
    'layer_size_max': 2048,
    'layer_size_step': 128, 
    'activation': {'type': 'categorical', 'choices': ['relu', 'elu']},
    'learning_rate': {'type': 'float', 'low': 1e-4, 'high': 1e-2, 'log': True},
    'weight_decay': {'type': 'float', 'low': 1e-6, 'high': 1e-4, 'log': True},
}


def estimate_optimal_batch_size(n_features, hidden_layers, available_memory_gb, safety_margin=0.7):
    """Estimate optimal batch size based on available GPU memory."""
    layers = [n_features] + hidden_layers + [1]  # 1 CV output
    memory_per_sample = (n_features + 2 * sum(hidden_layers)) * 4
    
    total_params = sum(layers[i] * layers[i+1] + layers[i+1] for i in range(len(layers) - 1))
    model_memory_gb = (total_params * 4 * 3) / 1e9
    usable_memory_gb = available_memory_gb * safety_margin - model_memory_gb
    
    if usable_memory_gb <= 0:
        return 16
    
    max_batch_size = int((usable_memory_gb * 1e9) / memory_per_sample)
    batch_size = max(16, min(max_batch_size, 4096))
    return 2 ** int(math.log2(batch_size))


def estimate_model_memory(n_features, hidden_layers, batch_size):
    """Estimate GPU memory required."""
    layers = [n_features] + hidden_layers + [1]  # 1 CV output
    total_params = sum(layers[i] * layers[i+1] + layers[i+1] for i in range(len(layers) - 1))
    
    param_memory = total_params * 4
    grad_memory = total_params * 4
    optimizer_memory = total_params * 8
    activation_memory = batch_size * (n_features + sum(hidden_layers)) * 4
    batch_grad_memory = batch_size * sum(hidden_layers) * 4
    
    total_memory_gb = (param_memory + grad_memory + optimizer_memory + 
                       activation_memory + batch_grad_memory) / 1e9
    return total_memory_gb, total_params


def load_full_dataset(filepath):
    """Load consolidated 3-state data."""
    print("\n" + "="*70)
    print("LOADING FULL DATASET")
    print("="*70)
    print(f"File: {filepath}")
    
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    print("Loading...")
    with np.load(filepath, allow_pickle=True) as npz_file:
        data = npz_file['data'].astype(np.float32)
        labels = npz_file['labels'].astype(np.int64)
    
    print(f"Data: {data.shape}, Memory: {data.nbytes / 1e9:.2f} GB")
    
    unique, counts = np.unique(labels, return_counts=True)
    print("\nClass distribution:")
    for cls, count, name in zip(unique, counts, ['Resting', 'Open', 'Desensitized']):
        print(f"  {name} ({cls}): {count:,} ({count/len(labels)*100:.1f}%)")
    
    return data, labels, data.shape[1]


def save_checkpoint(study, output_dir, checkpoint_file, config):
    """Save optimization checkpoint."""
    checkpoint_path = output_dir / checkpoint_file
    
    # Safely get best value
    try:
        best_value = study.best_value if study.best_trial else None
    except ValueError:
        best_value = None
    
    checkpoint_data = {
        'study': study,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'n_trials_complete': len([t for t in study.trials if t.state == TrialState.COMPLETE]),
        'n_trials_pruned': len([t for t in study.trials if t.state == TrialState.PRUNED]),
        'best_value': best_value,
        'config': config
    }
    
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    
    print(f"Checkpoint saved: {checkpoint_path}")


def load_checkpoint(output_dir, checkpoint_file):
    """Load optimization checkpoint if it exists."""
    checkpoint_path = output_dir / checkpoint_file
    
    if not checkpoint_path.exists():
        return None
    
    print("\n" + "="*70)
    print("RESUMING FROM CHECKPOINT")
    print("="*70)
    
    with open(checkpoint_path, 'rb') as f:
        checkpoint_data = pickle.load(f)
    
    print(f"Checkpoint loaded from: {checkpoint_path}")
    print(f"Saved at: {checkpoint_data['timestamp']}")
    print(f"Completed trials: {checkpoint_data['n_trials_complete']}")
    print(f"Pruned trials: {checkpoint_data['n_trials_pruned']}")
    if checkpoint_data['best_value']:
        print(f"Best value so far: {checkpoint_data['best_value']:.6f}")
    
    return checkpoint_data['study']


def objective(trial, data, labels, n_features, config):
    """Objective function with automatic batch size adjustment and early stopping."""
    
    # Get GPU memory
    if config['device'] == 'cuda':
        available_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    else:
        available_memory_gb = float('inf')
    
    # Sample hyperparameters
    n_layers = trial.suggest_int('n_layers', 
                                  HYPERPARAMETER_SPACE['n_layers']['low'],
                                  HYPERPARAMETER_SPACE['n_layers']['high'])
    
    hidden_layers = [trial.suggest_int(f'layer_{i}_size',
                                       HYPERPARAMETER_SPACE['layer_size_min'],
                                       HYPERPARAMETER_SPACE['layer_size_max'],
                                       step=HYPERPARAMETER_SPACE['layer_size_step'])
                    for i in range(n_layers)]
    
    activation = trial.suggest_categorical('activation', 
                                           HYPERPARAMETER_SPACE['activation']['choices'])
    learning_rate = trial.suggest_float('learning_rate',
                                        HYPERPARAMETER_SPACE['learning_rate']['low'],
                                        HYPERPARAMETER_SPACE['learning_rate']['high'],
                                        log=True)
    weight_decay = trial.suggest_float('weight_decay',
                                       HYPERPARAMETER_SPACE['weight_decay']['low'],
                                       HYPERPARAMETER_SPACE['weight_decay']['high'],
                                       log=True)
    
    # Auto-adjust batch size
    if config['device'] == 'cuda':
        batch_size = estimate_optimal_batch_size(n_features, hidden_layers, available_memory_gb)
    else:
        batch_size = config['batch_size']
    
    estimated_memory_gb, total_params = estimate_model_memory(n_features, hidden_layers, batch_size)
    
    # Get TDA parameters from config
    n_states = config['n_states']
    n_cvs = config['n_cvs']
    target_centers = config['target_centers']
    target_sigmas = config['target_sigmas']
    
    print(f"\n{'='*70}")
    print(f"Trial {trial.number}")
    print(f"{'='*70}")
    print(f"Architecture: {[n_features] + hidden_layers + [n_cvs]} → {n_cvs} CV for {n_states} states")
    print(f"Params: {total_params:,}, Est. memory: {estimated_memory_gb:.2f} GB")
    print(f"Batch size: {batch_size}, LR: {learning_rate:.6f}, Act: {activation}")
    
    # Check memory
    if config['device'] == 'cuda' and estimated_memory_gb > available_memory_gb * 0.95:
        print(f"Memory exceeds 95% of {available_memory_gb:.2f} GB. Pruning.")
        raise optuna.TrialPruned()
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=config['n_folds'], shuffle=True, 
                          random_state=config['random_seed'])
    device = config['device']
    fold_losses = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(data, labels)):
        print(f"\n--- Fold {fold + 1}/{config['n_folds']} ---")
        
        # Create datasets
        train_data = torch.from_numpy(data[train_idx]).float()
        train_labels = torch.from_numpy(labels[train_idx]).long()
        val_data = torch.from_numpy(data[val_idx]).float()
        val_labels = torch.from_numpy(labels[val_idx]).long()
        
        train_loader = DataLoader(TensorDataset(train_data, train_labels),
                                  batch_size=batch_size, shuffle=True,
                                  pin_memory=(device == 'cuda'), num_workers=0,
                                  drop_last=True)  # Drop incomplete batches
        val_loader = DataLoader(TensorDataset(val_data, val_labels),
                                batch_size=batch_size, shuffle=False,
                                pin_memory=(device == 'cuda'), num_workers=0,
                                drop_last=True)  # Drop incomplete batches to ensure all states present
        
        # Create model
        try:
            model = create_deep_tda_model(
                n_features=n_features,
                n_states=n_states,
                hidden_layers=hidden_layers,
                target_centers=target_centers,
                target_sigmas=target_sigmas,
                activation=activation,
                device=device,
                n_cvs=n_cvs
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"OOM during model creation. Pruning.")
                if device == 'cuda':
                    torch.cuda.empty_cache()
                raise optuna.TrialPruned()
            raise
        
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, 
                                     weight_decay=weight_decay)
        loss_fn = TDALoss(n_states=n_states, target_centers=target_centers, 
                         target_sigmas=target_sigmas)
        
        # Training with early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        
        try:
            for epoch in range(config['n_epochs_per_fold']):
                # Train
                model.train()
                train_loss = 0.0
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = loss_fn(model(batch_X), batch_y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * len(batch_X)
                train_loss /= len(train_data)
                
                # Validate
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                        try:
                            batch_loss = loss_fn(model(batch_X), batch_y).item() * len(batch_X)
                            val_loss += batch_loss
                        except ValueError as e:
                            # Skip batches that don't have all states (shouldn't happen with drop_last=True)
                            if "not represented in this batch" in str(e):
                                continue
                            raise
                
                # Only compute average if we processed any batches
                if val_loss > 0:
                    val_loss /= len(val_data)
                else:
                    # Fallback if all batches were skipped (very unlikely)
                    val_loss = train_loss
                
                # Early stopping check
                if val_loss < best_val_loss - config['early_stopping_min_delta']:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Stop if no improvement
                if patience_counter >= config['early_stopping_patience']:
                    print(f"-> Early stop at epoch {epoch+1} (no improvement for {patience_counter} epochs)")
                    break
                
                # Cleanup
                if (epoch + 1) % config['clear_cache_every_n_epochs'] == 0:
                    gc.collect()
                    if device == 'cuda':
                        torch.cuda.empty_cache()
                
                # Progress
                if (epoch + 1) % config['print_progress_every_n_epochs'] == 0 or epoch == 0:
                    print(f"  Epoch {epoch+1}: train={train_loss:.6f}, val={val_loss:.6f}, best={best_val_loss:.6f}")
                
                # Pruning - more aggressive
                if (epoch + 1) % config['pruning_check_interval'] == 0:
                    trial.report(val_loss, epoch)
                    if trial.should_prune():
                        print(f"  Pruned at epoch {epoch+1}")
                        raise optuna.TrialPruned()
        
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f" OOM during training. Pruning.")
                if device == 'cuda':
                    torch.cuda.empty_cache()
                raise optuna.TrialPruned()
            raise
        finally:
            # Always cleanup
            del model, optimizer, train_data, train_labels, val_data, val_labels
            gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()
        
        fold_losses.append(best_val_loss)
        print(f"Fold {fold + 1}: {best_val_loss:.6f}")
    
    mean_loss = np.mean(fold_losses)
    std_loss = np.std(fold_losses)
    print(f"\nTrial {trial.number}: {mean_loss:.6f} ± {std_loss:.6f}")
    return mean_loss


def main():
    config = OPTIMIZATION_CONFIG
    
    print("="*70)
    print("HYPERPARAMETER OPTIMIZATION - 3-STATE DeepTDA (FULL DATASET)")
    print("="*70)
    print(f"Training: {config['n_cvs']} CV to discriminate {config['n_states']} states")
    print(f"Target centers: {config['target_centers']}")
    print(f"Dataset: FULL (72,000 samples)")
    print(f"Optimization: {config['n_trials']} trials, {config['n_folds']}-fold CV, {config['n_epochs_per_fold']} epochs/fold")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup device
    if config['device'] == 'cuda' and not torch.cuda.is_available():
        print("\nCUDA not available, using CPU")
        config['device'] = 'cpu'
    elif config['device'] == 'cuda':
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print("Batch size: AUTO-ADJUSTED per trial")
    
    # Load data
    data, labels, n_features = load_full_dataset(config['data_file'])
    
    # Create output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Try to load checkpoint
    study = None
    if config['resume_from_checkpoint'] and config['enable_checkpointing']:
        study = load_checkpoint(output_dir, config['checkpoint_file'])
    
    # Create or use existing study
    if study is None:
        study_name = config['study_name'] or f"deeptda_3state_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print("\n" + "="*70)
        print("CREATING NEW OPTUNA STUDY")
        print("="*70)
        
        pruner = (optuna.pruners.MedianPruner(
                    n_warmup_steps=config['pruning_warmup_steps'],
                    n_startup_trials=5,  # Collect some baseline data first
                    interval_steps=config['pruning_check_interval']
                  ) if config['enable_pruning'] else optuna.pruners.NopPruner())
        
        study = optuna.create_study(
            study_name=study_name,
            direction='minimize',
            pruner=pruner,
            sampler=optuna.samplers.TPESampler(seed=config['random_seed'], n_startup_trials=10)
        )
        
        print(f"New study: {study_name}")
    else:
        study_name = study.study_name
        print(f"Resuming study: {study_name}")
    
    print(f"Config: {config['n_trials']} trials, {config['n_folds']}-fold CV, {config['n_epochs_per_fold']} max epochs")
    print(f"Early stopping: patience={config['early_stopping_patience']}")
    if config['enable_checkpointing']:
        print(f"Checkpointing: Every {config['checkpoint_freq_trials']} trials")
    print("="*70 + "\n")
    
    # Optimize
    start_time = time.time()
    timeout_seconds = config['timeout_hours'] * 3600 if config['timeout_hours'] else None
    
    remaining_trials = max(0, config['n_trials'] - len(study.trials))
    
    if remaining_trials == 0:
        print(f"All {config['n_trials']} trials completed!")
    else:
        print(f"Running {remaining_trials} remaining trials...\n")
        
        def checkpoint_callback(study, trial):
            if config['enable_checkpointing'] and trial.number % config['checkpoint_freq_trials'] == 0:
                save_checkpoint(study, output_dir, config['checkpoint_file'], config)
        
        try:
            study.optimize(
                lambda t: objective(t, data, labels, n_features, config),
                n_trials=remaining_trials,
                timeout=timeout_seconds,
                show_progress_bar=True,
                callbacks=[checkpoint_callback]
            )
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            if config['enable_checkpointing']:
                save_checkpoint(study, output_dir, config['checkpoint_file'], config)
        except Exception as e:
            print(f"\nError: {e}")
            if config['enable_checkpointing']:
                save_checkpoint(study, output_dir, config['checkpoint_file'], config)
            raise
        finally:
            if config['enable_checkpointing']:
                save_checkpoint(study, output_dir, config['checkpoint_file'], config)
    
    elapsed = time.time() - start_time
    
    # Results
    complete = [t for t in study.trials if t.state == TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == TrialState.PRUNED]
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Time: {elapsed/3600:.2f}h, Complete: {len(complete)}, Pruned: {len(pruned)}")
    print(f"Pruning efficiency: {len(pruned)/(len(complete)+len(pruned))*100:.1f}% trials pruned")
    
    if not complete:
        print("No trials completed!")
        return
    
    best = study.best_trial
    n_layers = best.params['n_layers']
    hidden = [best.params[f'layer_{i}_size'] for i in range(n_layers)]
    
    print(f"\nBest: Loss={best.value:.6f}")
    print(f"   Architecture: {hidden} → {config['n_cvs']} CV for {config['n_states']} states")
    print(f"   LR: {best.params['learning_rate']:.6f}")
    print(f"   Activation: {best.params['activation']}")
    print(f"   Weight Decay: {best.params['weight_decay']:.6f}")
    
    # Save results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    (output_dir / f"{study_name}_best.json").write_text(
        json.dumps({
            'loss': best.value,
            'params': best.params,
            'architecture': hidden,
            'n_cvs': config['n_cvs'],
            'n_states': config['n_states'],
            'target_centers': config['target_centers'],
            'target_sigmas': config['target_sigmas'],
            'n_trials': len(study.trials),
            'n_complete': len(complete),
            'n_pruned': len(pruned),
            'dataset_size': len(labels),
            'n_features': n_features,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }, indent=2)
    )
    
    study.trials_dataframe().to_csv(output_dir / f"{study_name}_trials.csv")
    
    print(f"Best config: {study_name}_best.json")
    print(f"All trials: {study_name}_trials.csv")
    print("="*70)


if __name__ == "__main__":
    main()