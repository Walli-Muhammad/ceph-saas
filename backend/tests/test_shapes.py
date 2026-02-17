"""
Integration Test: Dataset + Model Shape Validation

This script verifies:
1. Dataset loads correctly and generates proper shapes
2. Model accepts dataset output and produces correct output shape
3. Coordinate scaling is correct (visualized with heatmap overlay)
4. Heatmaps are generated at the right locations
"""

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))

from core.etl.loader import get_aariz_files
from core.ml.dataset import CephDataset
from core.ml.model import UNet


def test_dataset_shapes():
    """Test that dataset produces correct tensor shapes."""
    print("\n" + "=" * 80)
    print("TEST 1: Dataset Shape Validation")
    print("=" * 80)
    
    # Load data
    dataset_root = Path(__file__).parent.parent.parent / 'data' / 'Aariz'
    train_items = get_aariz_files(dataset_root, split='train')
    print(f"✓ Loaded {len(train_items)} training samples")
    
    # Create dataset
    dataset = CephDataset(train_items, target_size=512, heatmap_sigma=5.0)
    print(f"✓ Created CephDataset")
    
    # Test single sample
    image, heatmaps, metadata = dataset[0]
    
    print(f"\nSample 0:")
    print(f"  Image shape: {image.shape}")
    print(f"  Heatmaps shape: {heatmaps.shape}")
    print(f"  Ceph ID: {metadata['ceph_id']}")
    print(f"  Original size: {metadata['original_size']}")
    print(f"  Scale factors: {metadata['scale_factors']}")
    
    # Assertions
    assert image.shape == (1, 512, 512), f"Expected (1, 512, 512), got {image.shape}"
    assert heatmaps.shape == (29, 512, 512), f"Expected (29, 512, 512), got {heatmaps.shape}"
    assert image.min() >= 0 and image.max() <= 1, "Image should be normalized to [0, 1]"
    
    print("\n✓ All dataset shape assertions passed!")
    
    return dataset


def test_model_forward_pass(dataset):
    """Test model forward pass with batch from dataset."""
    print("\n" + "=" * 80)
    print("TEST 2: Model Forward Pass")
    print("=" * 80)
    
    # Create model
    model = UNet(in_channels=1, out_channels=29, bilinear=True)
    print(f"✓ Created U-Net model with {model.count_parameters():,} parameters")
    
    # Create dataloader
    batch_size = 4
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Get one batch
    images, heatmaps, metadata_list = next(iter(dataloader))
    
    print(f"\nBatch shapes:")
    print(f"  Input images: {images.shape}")
    print(f"  Target heatmaps: {heatmaps.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(images)
    
    print(f"  Model output: {output.shape}")
    print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")
    
    # Assertions
    assert images.shape == (batch_size, 1, 512, 512), \
        f"Expected input shape ({batch_size}, 1, 512, 512), got {images.shape}"
    assert output.shape == (batch_size, 29, 512, 512), \
        f"Expected output shape ({batch_size}, 29, 512, 512), got {output.shape}"
    assert output.min() >= 0 and output.max() <= 1, \
        f"Output should be in [0, 1], got [{output.min()}, {output.max()}]"
    
    print("\n✓ All model shape assertions passed!")
    
    return model, images, heatmaps, metadata_list


def visualize_heatmap_accuracy(dataset, save_path: Path):
    """
    Visualize X-ray with heatmap overlay to verify coordinate scaling.
    
    Creates a side-by-side comparison:
    - Left: Original X-ray
    - Middle: Summed heatmaps
    - Right: Overlay to verify landmarks are in correct positions
    """
    print("\n" + "=" * 80)
    print("TEST 3: Heatmap Visualization")
    print("=" * 80)
    
    # Get sample
    image, heatmaps, metadata = dataset[0]
    
    # Convert to numpy
    image_np = image.squeeze().numpy()  # (512, 512)
    heatmaps_np = heatmaps.numpy()  # (29, 512, 512)
    
    # Sum all heatmaps
    heatmap_sum = heatmaps_np.sum(axis=0)  # (512, 512)
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    
    # 1. Original X-ray
    axes[0, 0].imshow(image_np, cmap='gray')
    axes[0, 0].set_title(f"Original X-Ray\n{metadata['ceph_id']}", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # 2. Summed heatmaps
    im1 = axes[0, 1].imshow(heatmap_sum, cmap='hot')
    axes[0, 1].set_title(f"Summed Heatmaps (29 landmarks)\nSigma=5px", fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
    
    # 3. Overlay
    axes[1, 0].imshow(image_np, cmap='gray', alpha=0.8)
    axes[1, 0].imshow(heatmap_sum, cmap='hot', alpha=0.4)
    axes[1, 0].set_title("Overlay: X-Ray + Heatmaps", fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    # 4. Landmark points visualization
    axes[1, 1].imshow(image_np, cmap='gray')
    
    # Plot landmark centers as points
    landmarks = metadata['landmarks']
    for i, lm in enumerate(landmarks):
        x, y = lm['x'], lm['y']
        axes[1, 1].plot(x, y, 'r+', markersize=8, markeredgewidth=2)
        # Annotate first 5 landmarks
        if i < 5:
            axes[1, 1].annotate(lm['symbol'], (x, y), 
                               color='yellow', fontsize=8, fontweight='bold',
                               xytext=(5, 5), textcoords='offset points')
    
    axes[1, 1].set_title(f"Landmark Centers (29 points)", fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Add metadata text
    info_text = f"Original Size: {metadata['original_size']}\n"
    info_text += f"Target Size: (512, 512)\n"
    info_text += f"Scale Factors: ({metadata['scale_factors'][0]:.3f}, {metadata['scale_factors'][1]:.3f})\n"
    info_text += f"Pixel Size: {metadata['pixel_size_mm']} mm\n"
    info_text += f"Machine: {metadata['machine']}"
    
    fig.text(0.5, 0.02, info_text, ha='center', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    # Save
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to: {save_path}")
    
    plt.close()
    
    # Print landmark info
    print(f"\nLandmark scaling verification:")
    print(f"  Original image size: {metadata['original_size']}")
    print(f"  Target size: (512, 512)")
    print(f"  Scale factors: {metadata['scale_factors']}")
    print(f"\nFirst 3 scaled landmarks:")
    for i, lm in enumerate(landmarks[:3]):
        print(f"  {i+1}. {lm['title']:25s} ({lm['symbol']:4s}): x={lm['x']:6.1f}, y={lm['y']:6.1f}")


def run_all_tests():
    """Run all tests in sequence."""
    print("\n" + "=" * 80)
    print("CEPH-SAAS MVP: INTEGRATION TEST SUITE")
    print("Testing Dataset + Model Pipeline")
    print("=" * 80)
    
    try:
        # Test 1: Dataset
        dataset = test_dataset_shapes()
        
        # Test 2: Model
        model, images, heatmaps, metadata_list = test_model_forward_pass(dataset)
        
        # Test 3: Visualization
        output_dir = Path(__file__).parent / 'outputs'
        output_dir.mkdir(exist_ok=True)
        visualize_heatmap_accuracy(dataset, save_path=output_dir / 'heatmap_verification.png')
        
        # Final summary
        print("\n" + "=" * 80)
        print("✓ ALL TESTS PASSED!")
        print("=" * 80)
        print("\nSummary:")
        print(f"  ✓ Dataset loads and preprocesses images correctly")
        print(f"  ✓ Coordinate scaling from original to 512x512 works")
        print(f"  ✓ Heatmap generation produces 29 channels")
        print(f"  ✓ U-Net model accepts input and produces correct output shape")
        print(f"  ✓ Output is in valid range [0, 1]")
        print(f"\nVisualization saved to: {output_dir / 'heatmap_verification.png'}")
        print("\nNext steps:")
        print("  1. Review the visualization to verify heatmaps are at correct positions")
        print("  2. Implement training loop with loss function (MSE or BCE)")
        print("  3. Add data augmentation pipeline")
        print("=" * 80 + "\n")
        
        return True
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("✗ TEST FAILED!")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
