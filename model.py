"""
Model creation utilities for DeepTDA.

This module provides helper functions to create and configure DeepTDA models.
"""

import torch
from mlcolvar.cvs import DeepTDA


def create_deep_tda_model(n_features, n_states, hidden_layers, 
                          target_centers=None, target_sigmas=None,
                          activation='relu', device='cpu', n_cvs=1):
    """
    Create a DeepTDA model with specified architecture.
    
    Parameters:
    -----------
    n_features : int
        Number of input features
    n_states : int
        Number of states to classify (2 or 3)
    hidden_layers : list of int
        List of hidden layer sizes, e.g. [512, 256, 128]
    target_centers : list of float or None
        Target centers for each state.
    target_sigmas : list of float or None
        Target sigmas (widths) for each state.
    activation : str
        Activation function supported by mlcolvar:
        'relu', 'elu', 'tanh', 'softplus', 'shifted_softplus', 'linear'
    device : str
        Device to put model on: 'cpu' or 'cuda'
    n_cvs : int
        Number of collective variables (default: 1)
    
    Returns:
    --------
    model : DeepTDA
        Configured DeepTDA model on specified device
    """
    
    # Validate inputs
    if n_states not in [2, 3]:
        raise ValueError(f"n_states must be 2 or 3, got {n_states}")
    
    if not isinstance(hidden_layers, (list, tuple)) or len(hidden_layers) == 0:
        raise ValueError(f"hidden_layers must be a non-empty list, got {hidden_layers}")
    
    # Validate activation 
    valid_activations = ['relu', 'elu', 'tanh', 'softplus', 'shifted_softplus', 'linear']
    if activation not in valid_activations:
        raise ValueError(f"activation must be one of {valid_activations}, got '{activation}'")
    
    # Validate target_centers and target_sigmas 
    if target_centers is not None:
        if len(target_centers) != n_states:
            raise ValueError(f"target_centers length ({len(target_centers)}) must match n_states ({n_states})")
    
    if target_sigmas is not None:
        if len(target_sigmas) != n_states:
            raise ValueError(f"target_sigmas length ({len(target_sigmas)}) must match n_states ({n_states})")
    
    # Build layer architecture: input -> hidden layers -> output
    layers = [n_features] + list(hidden_layers) + [n_cvs]
    
    # Build options dictionary - activation goes inside 'nn' subdictionary
    options = {
        'nn': {
            'activation': activation
        }
    }
    
    # Create model - pass target centers and sigmas explicitly
    model = DeepTDA(
        layers=layers,
        n_states=n_states,
        target_centers=target_centers,
        target_sigmas=target_sigmas,
        n_cvs=n_cvs,
        options=options
    )
    
    # Move to device
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{'='*70}")
    print(f"DeepTDA Model Created")
    print(f"{'='*70}")
    print(f"Input features: {n_features:,}")
    print(f"Number of states: {n_states}")
    print(f"Number of CVs: {n_cvs}")
    print(f"Architecture: {layers}")
    print(f"Activation: {activation}")
    print(f"Target centers: {target_centers if target_centers else 'Not specified'}")
    print(f"Target sigmas: {target_sigmas if target_sigmas else 'Not specified'}")
    print(f"Device: {device}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"{'='*70}\n")
    
    return model


def load_trained_model(checkpoint_path, device='cpu'):
    """
    Load a trained DeepTDA model from checkpoint.
    
    Parameters:
    -----------
    checkpoint_path : str
        Path to saved model checkpoint (.pt file)
    device : str
        Device to load model on: 'cpu' or 'cuda'
    
    Returns:
    --------
    model : DeepTDA
        Loaded model
    checkpoint : dict
        Full checkpoint dictionary with training info
    """
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model parameters
    n_features = checkpoint['n_features']
    params = checkpoint['params']
    
    # Recreate model
    model = create_deep_tda_model(
        n_features=n_features,
        n_states=params.get('n_states', 2),
        hidden_layers=params['hidden_layers'],
        target_centers=params.get('target_centers'),
        target_sigmas=params.get('target_sigmas'),
        activation=params.get('activation', 'relu'),
        device=device,
        n_cvs=params.get('n_cvs', 1)
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from: {checkpoint_path}")
    
    return model, checkpoint


def count_parameters(model):
    """
    Count total and trainable parameters in model.
    
    Parameters:
    -----------
    model : torch.nn.Module
        PyTorch model
    
    Returns:
    --------
    total : int
        Total parameters
    trainable : int
        Trainable parameters
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_model_summary(model):
    """
    Print detailed model architecture summary.
    
    Parameters:
    -----------
    model : DeepTDA
        Model to summarize
    """
    print("\n" + "="*70)
    print("MODEL ARCHITECTURE SUMMARY")
    print("="*70)
    
    total, trainable = count_parameters(model)
    
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Non-trainable parameters: {total - trainable:,}")
    
    print("\nLayers:")
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                print(f"  {name}: {module.__class__.__name__} ({params:,} params)")
    
    print("="*70 + "\n")


if __name__ == "__main__":

    # Example usage for GH
    print("Example: Creating a 3-state DeepTDA model")
    
    model = create_deep_tda_model(
        n_features=1405326,
        n_states=3,
        hidden_layers=[512, 256, 128],
        target_centers=[-10.0, 0.0, 10.0],
        target_sigmas=[0.5, 0.5, 0.5],
        activation='relu',
        device='cpu'
    )
    
    print_model_summary(model)
    
     # Example usage for Input / OUTPUT
    print("\nExample input/output:")
    # Create dummy input
    x = torch.randn(4, 1405326)  # Batch of 4 samples
    with torch.no_grad():
        output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output (CV values):\n{output}")