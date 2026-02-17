"""
PyTorch Dataset for Aariz Cephalometric Dataset

This module implements the CephDataset class that:
1. Loads and preprocesses X-ray images (grayscale, resize to 512x512, normalize)
2. Scales landmark coordinates from original image size to 512x512
3. Generates 29-channel heatmap targets with Gaussian blobs for U-Net training
"""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Tuple, Dict
import logging

# Import our custom loader
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from core.etl.loader import AarizDatasetItem, load_annotation

logger = logging.getLogger(__name__)


class CephDataset(Dataset):
    """
    PyTorch Dataset for Cephalometric Landmark Detection.
    
    Loads X-ray images and generates heatmap targets for 29 anatomical landmarks.
    All images are resized to 512x512 for consistent model input.
    """
    
    def __init__(
        self,
        items: List[AarizDatasetItem],
        target_size: int = 512,
        heatmap_sigma: float = 5.0,
        transform=None
    ):
        """
        Initialize the dataset.
        
        Args:
            items: List of AarizDatasetItem from the loader
            target_size: Target image size (width and height)
            heatmap_sigma: Standard deviation for Gaussian heatmap blobs (in pixels)
            transform: Optional Albumentations transform pipeline
        """
        self.items = items
        self.target_size = target_size
        self.heatmap_sigma = heatmap_sigma
        self.transform = transform
        
        logger.info(f"Initialized CephDataset with {len(items)} samples")
        logger.info(f"Target size: {target_size}x{target_size}, Heatmap sigma: {heatmap_sigma}px")
    
    def __len__(self) -> int:
        return len(self.items)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample
        
        Returns:
            Tuple of (image_tensor, heatmap_tensor, metadata):
            - image_tensor: (1, 512, 512) normalized grayscale image
            - heatmap_tensor: (29, 512, 512) heatmap targets
            - metadata: Dictionary with ceph_id, original_size, etc.
        """
        item = self.items[idx]
        
        # Load image in grayscale
        image = cv2.imread(str(item.image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load image: {item.image_path}")
        
        # Store original dimensions
        original_height, original_width = image.shape
        
        # Load annotation
        annotation = load_annotation(item.annotation_path)
        landmarks = annotation.get('landmarks', [])
        
        if len(landmarks) != 29:
            logger.warning(f"Expected 29 landmarks, got {len(landmarks)} for {item.ceph_id}")
        
        # Resize image to target size
        image_resized = cv2.resize(image, (self.target_size, self.target_size), 
                                   interpolation=cv2.INTER_LINEAR)
        
        # Normalize to [0, 1]
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        # Scale landmark coordinates
        scale_x = self.target_size / original_width
        scale_y = self.target_size / original_height
        
        scaled_landmarks = []
        for landmark in landmarks:
            orig_x = landmark['value']['x']
            orig_y = landmark['value']['y']
            
            new_x = orig_x * scale_x
            new_y = orig_y * scale_y
            
            scaled_landmarks.append({
                'title': landmark['title'],
                'symbol': landmark['symbol'],
                'x': new_x,
                'y': new_y
            })
        
        # Generate heatmaps (29 channels, one per landmark)
        heatmaps = self._generate_heatmaps(scaled_landmarks, self.target_size, self.heatmap_sigma)
        
        # Convert to PyTorch tensors
        # Image: (H, W) -> (1, H, W)
        image_tensor = torch.from_numpy(image_normalized).unsqueeze(0)
        
        # Heatmaps: (29, H, W)
        heatmap_tensor = torch.from_numpy(heatmaps).float()
        
        # Metadata for debugging/visualization
        metadata = {
            'ceph_id': item.ceph_id,
            'original_size': (original_width, original_height),
            'scale_factors': (scale_x, scale_y),
            'pixel_size_mm': item.pixel_size,
            'machine': item.machine,
            'landmarks': scaled_landmarks
        }
        
        return image_tensor, heatmap_tensor, metadata
    
    def _generate_heatmaps(
        self,
        landmarks: List[Dict],
        size: int,
        sigma: float
    ) -> np.ndarray:
        """
        Generate Gaussian heatmaps for all landmarks.
        
        Args:
            landmarks: List of landmark dictionaries with 'x' and 'y' keys
            size: Heatmap size (width and height)
            sigma: Standard deviation of Gaussian blob
        
        Returns:
            Numpy array of shape (num_landmarks, size, size)
        """
        num_landmarks = len(landmarks)
        heatmaps = np.zeros((num_landmarks, size, size), dtype=np.float32)
        
        # Create coordinate grids
        x = np.arange(0, size)
        y = np.arange(0, size)
        xx, yy = np.meshgrid(x, y)
        
        for i, landmark in enumerate(landmarks):
            center_x = landmark['x']
            center_y = landmark['y']
            
            # Skip if landmark is out of bounds
            if center_x < 0 or center_x >= size or center_y < 0 or center_y >= size:
                logger.warning(f"Landmark {landmark['symbol']} out of bounds: ({center_x}, {center_y})")
                continue
            
            # Generate 2D Gaussian
            gaussian = np.exp(-((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2 * sigma ** 2))
            
            # Normalize to [0, 1] range
            if gaussian.max() > 0:
                gaussian = gaussian / gaussian.max()
            
            heatmaps[i] = gaussian
        
        return heatmaps


def visualize_sample(dataset: CephDataset, idx: int, save_path: Path = None):
    """
    Visualize a single sample from the dataset.
    
    Args:
        dataset: CephDataset instance
        idx: Sample index
        save_path: Optional path to save the visualization
    """
    import matplotlib.pyplot as plt
    
    image, heatmaps, metadata = dataset[idx]
    
    # Convert tensors to numpy for visualization
    image_np = image.squeeze().numpy()  # (512, 512)
    heatmaps_np = heatmaps.numpy()  # (29, 512, 512)
    
    # Sum all heatmaps
    heatmap_sum = heatmaps_np.sum(axis=0)  # (512, 512)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image_np, cmap='gray')
    axes[0].set_title(f"X-Ray Image\n{metadata['ceph_id']}")
    axes[0].axis('off')
    
    # Summed heatmaps
    axes[1].imshow(heatmap_sum, cmap='hot')
    axes[1].set_title(f"Summed Heatmaps (29 landmarks)")
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(image_np, cmap='gray', alpha=0.7)
    axes[2].imshow(heatmap_sum, cmap='hot', alpha=0.3)
    axes[2].set_title("Overlay")
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    """
    Test the dataset implementation.
    """
    from core.etl.loader import get_aariz_files
    
    # Load dataset
    dataset_root = Path(__file__).parent.parent.parent.parent / 'data' / 'Aariz'
    train_items = get_aariz_files(dataset_root, split='train')
    
    print(f"Loaded {len(train_items)} training samples")
    
    # Create dataset
    dataset = CephDataset(train_items, target_size=512, heatmap_sigma=5.0)
    
    # Test single sample
    print("\nTesting dataset[0]...")
    image, heatmaps, metadata = dataset[0]
    
    print(f"Image shape: {image.shape}")  # Should be (1, 512, 512)
    print(f"Heatmaps shape: {heatmaps.shape}")  # Should be (29, 512, 512)
    print(f"Ceph ID: {metadata['ceph_id']}")
    print(f"Original size: {metadata['original_size']}")
    print(f"Scale factors: {metadata['scale_factors']}")
    print(f"Number of landmarks: {len(metadata['landmarks'])}")
    
    # Visualize
    output_dir = Path(__file__).parent.parent.parent / 'tests' / 'outputs'
    output_dir.mkdir(parents=True, exist_ok=True)
    visualize_sample(dataset, 0, save_path=output_dir / 'dataset_sample.png')
    
    print(f"\n✓ Dataset test passed!")
