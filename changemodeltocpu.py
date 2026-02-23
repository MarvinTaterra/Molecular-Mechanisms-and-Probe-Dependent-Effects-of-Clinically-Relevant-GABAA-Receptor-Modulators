"""
Convert PyTorch model to CPU-only version for PLUMED.
This ensures the model won't try to use GPU and conflict with GROMACS.
"""

import torch
from pathlib import Path

INPUT_MODEL = "deeptda_3state_final.ptc"  
OUTPUT_MODEL = "deeptda_3state_final_cpu.ptc"  

def convert_to_cpu(input_path, output_path):
    """Load model and save as CPU-only version."""
    
    print("="*70)
    print("CONVERTING MODEL TO CPU-ONLY")
    print("="*70)
    
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    if not input_path.exists():
        print(f"Error: Input model not found: {input_path}")
        return False
    
    print(f"Loading model from: {input_path}")
    
    try:
        model = torch.jit.load(str(input_path))
        print("Model loaded successfully")
        
        model = model.cpu()
        print("Model moved to CPU")
        model.eval()
        
        print("\nTesting model...")
        n_inputs = 10000 
        test_input = torch.randn(1, n_inputs)
        
        with torch.no_grad():
            test_output = model(test_input)
            print(f"Test successful: input shape {test_input.shape} -> output shape {test_output.shape}")
        
        print(f"\nSaving CPU-only model to: {output_path}")
        torch.jit.save(model, str(output_path))
        print("Model saved successfully")
        
        print("\nVerifying saved model...")
        loaded_model = torch.jit.load(str(output_path))
        loaded_model.eval()
        
        with torch.no_grad():
            verify_output = loaded_model(test_input)
            max_diff = torch.max(torch.abs(test_output - verify_output)).item()
            
        print(f"Verification complete: max difference = {max_diff:.2e}")
        
        if max_diff < 1e-6:
            print("Model conversion successful!")
        else:
            print(f"Warning: Non-negligible difference ({max_diff:.2e})")
        
        input_size = input_path.stat().st_size / (1024**2)
        output_size = output_path.stat().st_size / (1024**2)
        print(f"\n File sizes:")
        print(f"   Original: {input_size:.2f} MB")
        print(f"   CPU-only: {output_size:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    torch.set_default_device('cpu')
    
    success = convert_to_cpu(INPUT_MODEL, OUTPUT_MODEL)
    
    if success:
        print("\nConversion complete!")
        print(f"Use {OUTPUT_MODEL} in your PLUMED simulations")
    else:
        print("\nConversion failed!")