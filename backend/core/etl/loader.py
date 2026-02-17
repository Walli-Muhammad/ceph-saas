"""
Aariz Dataset Loader Module

This module handles loading and validating the Aariz cephalometric dataset.
It pairs images with their corresponding Senior Orthodontist annotations and
reads pixel resolution metadata from the CSV file.

Dataset Structure:
- Images: data/Aariz/{train|test|valid}/Cephalograms/
- Annotations: data/Aariz/{train|test|valid}/Annotations/Cephalometric Landmarks/Senior Orthodontists/
- Metadata: data/Aariz/cephalogram_machine_mappings.csv
"""

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AarizDatasetItem:
    """Represents a single item in the Aariz dataset."""
    
    def __init__(
        self,
        ceph_id: str,
        image_path: Path,
        annotation_path: Path,
        pixel_size: float,
        image_format: str,
        machine: str
    ):
        self.ceph_id = ceph_id
        self.image_path = image_path
        self.annotation_path = annotation_path
        self.pixel_size = pixel_size  # Physical pixel size in mm
        self.image_format = image_format
        self.machine = machine
    
    def __repr__(self) -> str:
        return f"AarizDatasetItem(ceph_id={self.ceph_id}, format={self.image_format}, pixel_size={self.pixel_size}mm)"


def load_resolution_metadata(csv_path: Path) -> Dict[str, Dict[str, str]]:
    """
    Load pixel resolution metadata from the CSV file.
    
    Args:
        csv_path: Path to cephalogram_machine_mappings.csv
    
    Returns:
        Dictionary mapping cephalogram_id to metadata dict with keys:
        - machine: X-ray machine name
        - pixel_size: Physical pixel size in mm
        - image_format: Image file format (jpg, png, bmp, etc.)
        - mode: Dataset split (Train, Test, Valid)
    """
    metadata = {}
    
    if not csv_path.exists():
        logger.error(f"Metadata CSV not found: {csv_path}")
        return metadata
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ceph_id = row['cephalogram_id']
                metadata[ceph_id] = {
                    'machine': row['machine'],
                    'pixel_size': float(row['pixel_size']),
                    'image_format': row['image_format'],
                    'mode': row['mode']
                }
        
        logger.info(f"Loaded metadata for {len(metadata)} cephalograms")
    except Exception as e:
        logger.error(f"Error loading metadata CSV: {e}")
    
    return metadata


def find_image_file(ceph_id: str, cephalograms_dir: Path, expected_format: str) -> Optional[Path]:
    """
    Find the image file for a given cephalogram ID.
    
    Args:
        ceph_id: Cephalogram identifier
        cephalograms_dir: Directory containing cephalogram images
        expected_format: Expected file format from metadata
    
    Returns:
        Path to image file if found, None otherwise
    """
    # Try the expected format first
    image_path = cephalograms_dir / f"{ceph_id}.{expected_format}"
    if image_path.exists():
        return image_path
    
    # Try common image extensions as fallback
    for ext in ['jpg', 'jpeg', 'png', 'bmp']:
        image_path = cephalograms_dir / f"{ceph_id}.{ext}"
        if image_path.exists():
            if ext != expected_format:
                logger.warning(f"Image format mismatch for {ceph_id}: expected {expected_format}, found {ext}")
            return image_path
    
    return None


def get_aariz_files(root_dir: Path, split: str = 'train') -> List[AarizDatasetItem]:
    """
    Scan the Aariz dataset directory and pair images with Senior Orthodontist annotations.
    
    Args:
        root_dir: Root directory of the Aariz dataset (e.g., data/Aariz/)
        split: Dataset split to load ('train', 'test', or 'valid')
    
    Returns:
        List of AarizDatasetItem objects containing paired data
    """
    root_dir = Path(root_dir)
    split_dir = root_dir / split
    
    # Define paths
    cephalograms_dir = split_dir / 'Cephalograms'
    senior_annotations_dir = split_dir / 'Annotations' / 'Cephalometric Landmarks' / 'Senior Orthodontists'
    metadata_csv = root_dir / 'cephalogram_machine_mappings.csv'
    
    # Validate directories exist
    if not split_dir.exists():
        logger.error(f"Split directory not found: {split_dir}")
        return []
    
    if not cephalograms_dir.exists():
        logger.error(f"Cephalograms directory not found: {cephalograms_dir}")
        return []
    
    if not senior_annotations_dir.exists():
        logger.error(f"Senior annotations directory not found: {senior_annotations_dir}")
        return []
    
    # Load metadata
    metadata = load_resolution_metadata(metadata_csv)
    
    # Scan for annotation files (JSON only, as per dataset structure)
    annotation_files = list(senior_annotations_dir.glob('*.json'))
    logger.info(f"Found {len(annotation_files)} Senior Orthodontist annotation files in {split}")
    
    dataset_items = []
    missing_images = []
    missing_metadata = []
    
    for annotation_path in annotation_files:
        # Extract cephalogram ID from filename (without extension)
        ceph_id = annotation_path.stem
        
        # Check if metadata exists
        if ceph_id not in metadata:
            logger.warning(f"No metadata found for {ceph_id}, skipping")
            missing_metadata.append(ceph_id)
            continue
        
        meta = metadata[ceph_id]
        
        # Find corresponding image file
        image_path = find_image_file(ceph_id, cephalograms_dir, meta['image_format'])
        
        if image_path is None:
            logger.warning(f"No image found for annotation {ceph_id}, skipping")
            missing_images.append(ceph_id)
            continue
        
        # Create dataset item
        item = AarizDatasetItem(
            ceph_id=ceph_id,
            image_path=image_path,
            annotation_path=annotation_path,
            pixel_size=meta['pixel_size'],
            image_format=meta['image_format'],
            machine=meta['machine']
        )
        dataset_items.append(item)
    
    # Log summary
    logger.info(f"Successfully paired {len(dataset_items)} images with annotations")
    if missing_images:
        logger.warning(f"Missing images for {len(missing_images)} annotations")
    if missing_metadata:
        logger.warning(f"Missing metadata for {len(missing_metadata)} annotations")
    
    return dataset_items


def load_annotation(annotation_path: Path) -> Dict:
    """
    Load and parse a JSON annotation file.
    
    Args:
        annotation_path: Path to the annotation JSON file
    
    Returns:
        Dictionary containing annotation data with keys:
        - ceph_id: Cephalogram identifier
        - landmarks: List of landmark dictionaries with x, y coordinates
        - reviewed_at: Review timestamp
        - updated_at: Update timestamp
    """
    try:
        with open(annotation_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Error loading annotation {annotation_path}: {e}")
        return {}


if __name__ == "__main__":
    """
    Main execution block for testing the data loader.
    """
    import sys
    
    # Define dataset root (relative to this file or absolute)
    # Assuming script is in backend/core/etl/ and data is in data/Aariz/
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent  # Go up to project root
    dataset_root = project_root / 'data' / 'Aariz'
    
    print("=" * 80)
    print("AARIZ DATASET LOADER - VERIFICATION")
    print("=" * 80)
    print(f"Dataset Root: {dataset_root}")
    print(f"Exists: {dataset_root.exists()}")
    print()
    
    # Scan the training set
    print("Scanning TRAIN split...")
    print("-" * 80)
    train_items = get_aariz_files(dataset_root, split='train')
    
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total Images Found: {len(train_items)}")
    
    if train_items:
        # Count valid annotations (those with landmarks)
        valid_count = 0
        for item in train_items:
            annotation = load_annotation(item.annotation_path)
            if annotation and 'landmarks' in annotation:
                valid_count += 1
        
        print(f"Total Valid Annotations: {valid_count}")
        
        # Print sample annotation content (first 3 landmarks)
        print()
        print("Sample Annotation Content (First Item):")
        print("-" * 80)
        sample_item = train_items[0]
        sample_annotation = load_annotation(sample_item.annotation_path)
        
        print(f"Cephalogram ID: {sample_annotation.get('ceph_id', 'N/A')}")
        print(f"Number of Landmarks: {len(sample_annotation.get('landmarks', []))}")
        print(f"Pixel Size: {sample_item.pixel_size} mm")
        print(f"Machine: {sample_item.machine}")
        print()
        print("First 3 Landmarks:")
        for i, landmark in enumerate(sample_annotation.get('landmarks', [])[:3], 1):
            print(f"  {i}. {landmark['title']} ({landmark['symbol']}): "
                  f"x={landmark['value']['x']}, y={landmark['value']['y']}")
    else:
        print("No valid dataset items found!")
        sys.exit(1)
    
    print()
    print("=" * 80)
    print("Dataset verification complete!")
    print("=" * 80)
