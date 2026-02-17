"""
Diagnostic script to debug NaN loss issue.
Checks for problems in data loading, heatmap generation, and model forward pass.
"""

import torch
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from core.etl.loader import get_aariz_files
from core.ml.dataset import CephDataset
from core.ml.model import UNet

def check_dataset():
    """Check if dataset produces valid data."""
    print("=" * 80)
    print("DATASET DIAGNOSTICS")
    print("=" * 80)
    
    # Load data
    dataset_root = Path(__file__).parent.parent.parent / 'data' / 'Aariz'
    train_items = get_aariz_files(dataset_root, split='train')[:10]  # Just 10 samples
    
    dataset = CephDataset(train_items, target_size=512, heatmap_sigma=5.0)
    
    print(f"\nChecking {len(dataset)} samples...")
    
    issues_found = []
    
    for i in range(len(dataset)):
        image, heatmaps, metadata = dataset[i]
        
        # Check for NaN or Inf
        if torch.isnan(image).any():
            issues_found.append(f"Sample {i}: Image contains NaN")
        if torch.isinf(image).any():
            issues_found.append(f"Sample {i}: Image contains Inf")
        if torch.isnan(heatmaps).any():
            issues_found.append(f"Sample {i}: Heatmaps contain NaN")
        if torch.isinf(heatmaps).any():
            issues_found.append(f"Sample {i}: Heatmaps contain Inf")
        
        # Check ranges
        if image.min() < 0 or image.max() > 1:
            issues_found.append(f"Sample {i}: Image out of range [0,1]: [{image.min():.4f}, {image.max():.4f}]")
        if heatmaps.min() < 0 or heatmaps.max() > 1:
            issues_found.append(f"Sample {i}: Heatmaps out of range [0,1]: [{heatmaps.min():.4f}, {heatmaps.max():.4f}]")
        
        # Check shapes
        if image.shape != (1, 512, 512):
            issues_found.append(f"Sample {i}: Wrong image shape: {image.shape}")
        if heatmaps.shape != (29, 512, 512):
            issues_found.append(f"Sample {i}: Wrong heatmaps shape: {heatmaps.shape}")
        
        print(f"Sample {i}: Image [{image.min():.4f}, {image.max():.4f}], Heatmaps [{heatmaps.min():.4f}, {heatmaps.max():.4f}]")
    
    if issues_found:
        print("\n❌ ISSUES FOUND:")
        for issue in issues_found:
            print(f"  - {issue}")
        return False
    else:
        print("\n✓ All dataset samples are valid!")
        return True

def check_model():
    """Check if model produces valid outputs."""
    print("\n" + "=" * 80)
    print("MODEL DIAGNOSTICS")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model = UNet(in_channels=1, out_channels=29).to(device)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(2, 1, 512, 512).to(device)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Input range: [{dummy_input.min():.4f}, {dummy_input.max():.4f}]")
    
    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    
    # Check for issues
    issues = []
    if torch.isnan(output).any():
        issues.append("Output contains NaN")
    if torch.isinf(output).any():
        issues.append("Output contains Inf")
    
    if issues:
        print("\n❌ MODEL ISSUES:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("\n✓ Model produces valid outputs!")
        return True

def check_loss_calculation():
    """Check if loss calculation produces NaN."""
    print("\n" + "=" * 80)
    print("LOSS CALCULATION DIAGNOSTICS")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load real data
    dataset_root = Path(__file__).parent.parent.parent / 'data' / 'Aariz'
    train_items = get_aariz_files(dataset_root, split='train')[:2]
    dataset = CephDataset(train_items, target_size=512, heatmap_sigma=5.0)
    
    # Get one sample
    image, heatmaps, metadata = dataset[0]
    image = image.unsqueeze(0).to(device)  # Add batch dimension
    heatmaps = heatmaps.unsqueeze(0).to(device)
    
    # Create model
    model = UNet(in_channels=1, out_channels=29).to(device)
    criterion = torch.nn.MSELoss()
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        predictions = model(image)
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"Targets range: [{heatmaps.min():.4f}, {heatmaps.max():.4f}]")
    
    # Calculate loss
    loss = criterion(predictions, heatmaps)
    
    print(f"\nLoss value: {loss.item()}")
    
    if torch.isnan(loss):
        print("❌ Loss is NaN!")
        return False
    elif torch.isinf(loss):
        print("❌ Loss is Inf!")
        return False
    else:
        print("✓ Loss is valid!")
        return True

def main():
    print("\n🔍 DEBUGGING NaN LOSS ISSUE\n")
    
    dataset_ok = check_dataset()
    model_ok = check_model()
    loss_ok = check_loss_calculation()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Dataset: {'✓ OK' if dataset_ok else '❌ FAILED'}")
    print(f"Model: {'✓ OK' if model_ok else '❌ FAILED'}")
    print(f"Loss Calculation: {'✓ OK' if loss_ok else '❌ FAILED'}")
    
    if not (dataset_ok and model_ok and loss_ok):
        print("\n⚠️  Issues detected! Review the diagnostics above.")
    else:
        print("\n✓ All checks passed! The NaN issue may be related to training dynamics.")
        print("\nPossible causes:")
        print("  1. Learning rate too high")
        print("  2. Gradient explosion during backprop")
        print("  3. AMP numerical instability")
        print("\nRecommended fixes:")
        print("  1. Add gradient clipping")
        print("  2. Reduce learning rate to 1e-5")
        print("  3. Disable AMP temporarily")
    
    print("=" * 80)

if __name__ == "__main__":
    main()
