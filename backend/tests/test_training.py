"""
Quick test script to verify training pipeline works.
Runs 2 epochs with small batch to test the pipeline.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import sys

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))

from core.etl.loader import get_aariz_files
from core.ml.dataset import CephDataset
from core.ml.model import UNet

def quick_test():
    """Quick test of training pipeline."""
    print("=" * 80)
    print("QUICK TRAINING TEST (2 epochs)")
    print("=" * 80)
    
    # Paths
    project_root = Path(__file__).parent.parent.parent
    dataset_root = project_root / 'data' / 'Aariz'
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load small subset
    print("\nLoading data...")
    train_items = get_aariz_files(dataset_root, split='train')[:20]  # Only 20 samples
    val_items = get_aariz_files(dataset_root, split='valid')[:10]    # Only 10 samples
    
    print(f"Train samples: {len(train_items)}")
    print(f"Val samples: {len(val_items)}")
    
    # Create datasets
    train_dataset = CephDataset(train_items, target_size=512, heatmap_sigma=5.0)
    val_dataset = CephDataset(val_items, target_size=512, heatmap_sigma=5.0)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    
    # Create model
    print("\nCreating model...")
    model = UNet(in_channels=1, out_channels=29).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Quick training loop
    print("\nTraining for 2 epochs...")
    print("-" * 80)
    
    for epoch in range(1, 3):
        # Train
        model.train()
        train_loss = 0.0
        for images, heatmaps, _ in train_loader:
            images, heatmaps = images.to(device), heatmaps.to(device)
            
            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, heatmaps)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, heatmaps, _ in val_loader:
                images, heatmaps = images.to(device), heatmaps.to(device)
                predictions = model(images)
                loss = criterion(predictions, heatmaps)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch}/2 | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    print("-" * 80)
    print("✓ Training pipeline test passed!")
    print("=" * 80)

if __name__ == "__main__":
    quick_test()
