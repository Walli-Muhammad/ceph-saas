"""
Visualize Model Predictions vs Ground Truth

This script loads the trained model and visualizes its predictions
on a test image, comparing predicted landmarks with ground truth.
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import sys
import random

sys.path.append(str(Path(__file__).parent.parent))

from core.etl.loader import get_aariz_files, load_annotation
from core.ml.dataset import CephDataset
from core.ml.model import UNet


def find_heatmap_peak(heatmap: np.ndarray) -> tuple:
    """
    Find the (x, y) coordinates of the brightest spot in a heatmap.
    
    Args:
        heatmap: 2D numpy array (512, 512)
    
    Returns:
        (x, y) coordinates of the peak
    """
    # Find the index of the maximum value
    max_idx = np.argmax(heatmap)
    y, x = np.unravel_index(max_idx, heatmap.shape)
    return int(x), int(y)


def visualize_predictions(
    model: torch.nn.Module,
    dataset: CephDataset,
    sample_idx: int,
    device: torch.device,
    save_path: Path
):
    """
    Visualize model predictions vs ground truth.
    
    Args:
        model: Trained U-Net model
        dataset: CephDataset instance
        sample_idx: Index of sample to visualize
        device: Device (cuda/cpu)
        save_path: Path to save visualization
    """
    # Get sample
    image, target_heatmaps, metadata = dataset[sample_idx]
    
    print(f"\nVisualizing sample: {metadata['ceph_id']}")
    print(f"Original size: {metadata['original_size']}")
    print(f"Number of landmarks: {len(metadata['landmarks'])}")
    
    # Run inference
    model.eval()
    with torch.no_grad():
        image_batch = image.unsqueeze(0).to(device)  # Add batch dimension
        predicted_heatmaps = model(image_batch)
        predicted_heatmaps = predicted_heatmaps.squeeze(0).cpu().numpy()  # (29, 512, 512)
    
    # Convert image to numpy for visualization
    image_np = image.squeeze().numpy()  # (512, 512)
    
    # Convert grayscale to RGB for colored annotations
    image_rgb = cv2.cvtColor((image_np * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    
    # Extract predicted landmark positions
    predicted_landmarks = []
    for i in range(29):
        x, y = find_heatmap_peak(predicted_heatmaps[i])
        predicted_landmarks.append((x, y))
    
    # Get ground truth landmarks
    gt_landmarks = [(int(lm['x']), int(lm['y'])) for lm in metadata['landmarks']]
    
    # Draw ground truth (Green circles)
    for i, (x, y) in enumerate(gt_landmarks):
        cv2.circle(image_rgb, (x, y), 3, (0, 255, 0), -1)  # Green filled circle
        cv2.circle(image_rgb, (x, y), 5, (0, 255, 0), 1)   # Green outline
    
    # Draw predictions (Red crosses)
    for i, (x, y) in enumerate(predicted_landmarks):
        cv2.drawMarker(image_rgb, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 8, 2)  # Red cross
    
    # Calculate Mean Radial Error (MRE)
    errors = []
    for (pred_x, pred_y), (gt_x, gt_y) in zip(predicted_landmarks, gt_landmarks):
        error = np.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
        errors.append(error)
    
    mre = np.mean(errors)
    max_error = np.max(errors)
    
    # Add text overlay with metrics
    cv2.putText(image_rgb, f"Mean Radial Error: {mre:.2f} px", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(image_rgb, f"Max Error: {max_error:.2f} px", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(image_rgb, "Green = Ground Truth | Red = Prediction", (10, image_rgb.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Save visualization
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), image_rgb)
    
    print(f"\n✓ Visualization saved to: {save_path}")
    print(f"\nMetrics:")
    print(f"  Mean Radial Error: {mre:.2f} pixels")
    print(f"  Max Error: {max_error:.2f} pixels")
    print(f"  Min Error: {np.min(errors):.2f} pixels")
    
    # Print per-landmark errors
    print(f"\nPer-Landmark Errors:")
    for i, (error, lm) in enumerate(zip(errors, metadata['landmarks'])):
        print(f"  {i+1:2d}. {lm['title']:30s} ({lm['symbol']:4s}): {error:6.2f} px")
    
    return mre, max_error


def main():
    """
    Main function to visualize model predictions.
    """
    print("=" * 80)
    print("MODEL PREDICTION VISUALIZATION")
    print("=" * 80)
    
    # Paths
    project_root = Path(__file__).parent.parent.parent
    dataset_root = project_root / 'data' / 'Aariz'
    checkpoint_path = project_root / 'backend' / 'checkpoints' / 'best_model.pth'
    output_path = project_root / 'backend' / 'tests' / 'outputs' / 'prediction_result.jpg'
    
    # Check if checkpoint exists
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Please train the model first!")
        return
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load test dataset
    print("\nLoading test dataset...")
    test_items = get_aariz_files(dataset_root, split='test')
    
    if len(test_items) == 0:
        print("No test samples found, using validation set...")
        test_items = get_aariz_files(dataset_root, split='valid')
    
    print(f"Found {len(test_items)} test samples")
    
    # Create dataset
    dataset = CephDataset(test_items, target_size=512, heatmap_sigma=5.0)
    
    # Pick a random sample
    sample_idx = random.randint(0, len(dataset) - 1)
    print(f"Selected random sample index: {sample_idx}")
    
    # Load model
    print("\nLoading trained model...")
    model = UNet(in_channels=1, out_channels=29, bilinear=True).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"  Best validation loss: {checkpoint.get('best_val_loss', 'N/A')}")
    
    # Visualize predictions
    print("\nGenerating visualization...")
    mre, max_error = visualize_predictions(model, dataset, sample_idx, device, output_path)
    
    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE!")
    print("=" * 80)
    print(f"\nOutput saved to: {output_path}")
    print(f"\nModel Performance:")
    print(f"  Mean Radial Error: {mre:.2f} pixels")
    print(f"  Max Error: {max_error:.2f} pixels")
    
    if mre < 2.0:
        print("\n✓ Excellent! MRE < 2.0 pixels")
    elif mre < 4.0:
        print("\n✓ Good! MRE < 4.0 pixels")
    else:
        print("\n⚠️  MRE is high, model may need more training")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
