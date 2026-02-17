"""
Training Pipeline for Cephalometric Landmark Detection
Optimized for NVIDIA GTX 1650 Ti (4GB VRAM)

Features:
- Automatic Mixed Precision (AMP) for memory efficiency
- Small batch size (2) to fit in 4GB VRAM
- Train/Validation split with proper checkpointing
- Progress tracking with tqdm
- Resume training from checkpoint (power failure recovery)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
from tqdm import tqdm
import sys
import logging
import argparse

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.etl.loader import get_aariz_files
from core.ml.dataset import CephDataset
from core.ml.model import UNet

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Trainer:
    """
    Training manager for U-Net landmark detection model.
    Optimized for GTX 1650 Ti with AMP support.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        checkpoint_dir: Path,
        num_epochs: int = 50,
        use_amp: bool = True,
        start_epoch: int = 1
    ):
        """
        Initialize trainer.
        
        Args:
            model: U-Net model
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function (MSE)
            optimizer: Optimizer (Adam)
            device: Device to train on (cuda)
            checkpoint_dir: Directory to save checkpoints
            num_epochs: Number of training epochs
            use_amp: Use Automatic Mixed Precision (recommended for GTX 1650 Ti)
            start_epoch: Starting epoch (for resume training)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.num_epochs = num_epochs
        self.use_amp = use_amp
        self.start_epoch = start_epoch
        
        # Initialize AMP GradScaler for mixed precision training
        self.scaler = GradScaler() if use_amp else None
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Track best validation loss
        self.best_val_loss = float('inf')
        
        logger.info(f"Trainer initialized on device: {device}")
        logger.info(f"AMP enabled: {use_amp}")
        logger.info(f"Checkpoints will be saved to: {checkpoint_dir}")
    
    def train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch with AMP support.
        
        Args:
            epoch: Current epoch number
        
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.num_epochs} [Train]")
        
        for batch_idx, (images, heatmaps, metadata) in enumerate(pbar):
            # Move to device
            images = images.to(self.device)
            heatmaps = heatmaps.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with AMP
            if self.use_amp:
                with autocast():
                    predictions = self.model(images)
                    loss = self.criterion(predictions, heatmaps)
                
                # Backward pass with gradient scaling and clipping
                self.scaler.scale(loss).backward()
                
                # Unscale gradients before clipping
                self.scaler.unscale_(self.optimizer)
                
                # Gradient clipping to prevent explosion
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard forward/backward pass
                predictions = self.model(images)
                loss = self.criterion(predictions, heatmaps)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
            
            # Track loss
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self, epoch: int) -> float:
        """
        Validate the model.
        
        Args:
            epoch: Current epoch number
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        # Progress bar
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch}/{self.num_epochs} [Val]  ")
        
        with torch.no_grad():
            for images, heatmaps, metadata in pbar:
                # Move to device
                images = images.to(self.device)
                heatmaps = heatmaps.to(self.device)
                
                # Forward pass (AMP also works in eval mode)
                if self.use_amp:
                    with autocast():
                        predictions = self.model(images)
                        loss = self.criterion(predictions, heatmaps)
                else:
                    predictions = self.model(images)
                    loss = self.criterion(predictions, heatmaps)
                
                # Track loss
                total_loss += loss.item()
                
                # Update progress bar
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def save_checkpoint(self, epoch: int, train_loss: float, val_loss: float, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            train_loss: Training loss
            val_loss: Validation loss
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'scaler_state_dict': self.scaler.state_dict() if self.use_amp else None
        }
        
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"✓ Saved best model to {best_path}")
    
    def train(self):
        """
        Main training loop.
        """
        logger.info("=" * 80)
        logger.info("Starting Training")
        logger.info("=" * 80)
        logger.info(f"Epochs: {self.num_epochs}")
        logger.info(f"Train batches: {len(self.train_loader)}")
        logger.info(f"Val batches: {len(self.val_loader)}")
        logger.info(f"Mixed Precision: {self.use_amp}")
        logger.info(f"Starting from epoch: {self.start_epoch}")
        logger.info("=" * 80)
        
        for epoch in range(self.start_epoch, self.num_epochs + 1):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate(epoch)
            
            # Check if this is the best model
            is_best = val_loss < self.best_val_loss
            
            # Print epoch summary
            if is_best:
                print(f"\nEpoch {epoch}/{self.num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | ✓ Saved Best Model!")
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, train_loss, val_loss, is_best=True)
            else:
                print(f"\nEpoch {epoch}/{self.num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            print("-" * 80)
        
        # Save final model
        final_checkpoint = {
            'epoch': self.num_epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'scaler_state_dict': self.scaler.state_dict() if self.use_amp else None
        }
        final_path = self.checkpoint_dir / 'final_model.pth'
        torch.save(final_checkpoint, final_path)
        logger.info(f"✓ Saved final model to {final_path}")
        
        logger.info("=" * 80)
        logger.info("Training Complete!")
        logger.info(f"Best Validation Loss: {self.best_val_loss:.4f}")
        logger.info("=" * 80)


def main():
    """
    Main training script optimized for GTX 1650 Ti (4GB VRAM).
    Supports resume training with --resume flag.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train U-Net for cephalometric landmark detection')
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
    args = parser.parse_args()
    
    # Hyperparameters (Optimized for 4GB VRAM)
    BATCH_SIZE = 2          # Small batch to fit in 4GB VRAM
    LEARNING_RATE = 5e-5    # Reduced for stability
    NUM_EPOCHS = 50
    NUM_WORKERS = 2         # For Windows compatibility
    TARGET_SIZE = 512
    HEATMAP_SIGMA = 5.0
    USE_AMP = False         # Disabled temporarily due to NaN loss issue
    
    # Paths
    project_root = Path(__file__).parent.parent.parent.parent
    dataset_root = project_root / 'data' / 'Aariz'
    checkpoint_dir = project_root / 'backend' / 'checkpoints'
    
    # Device - STRICTLY check for CUDA
    if not torch.cuda.is_available():
        logger.error("CUDA is not available! This script requires a GPU.")
        logger.error("Please ensure NVIDIA drivers and CUDA are properly installed.")
        sys.exit(1)
    
    device = torch.device('cuda')
    logger.info(f"Using device: {device}")
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    
    # Load datasets
    logger.info("\nLoading datasets...")
    train_items = get_aariz_files(dataset_root, split='train')
    val_items = get_aariz_files(dataset_root, split='valid')
    
    logger.info(f"Train samples: {len(train_items)}")
    logger.info(f"Validation samples: {len(val_items)}")
    
    # Create datasets
    train_dataset = CephDataset(train_items, target_size=TARGET_SIZE, heatmap_sigma=HEATMAP_SIGMA)
    val_dataset = CephDataset(val_items, target_size=TARGET_SIZE, heatmap_sigma=HEATMAP_SIGMA)
    
    # Create dataloaders with pin_memory for faster GPU transfer
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True  # Faster GPU transfer
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    # Create model
    logger.info("\nInitializing model...")
    model = UNet(in_channels=1, out_channels=29, bilinear=True)
    logger.info(f"Model parameters: {model.count_parameters():,}")
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    logger.info(f"Loss function: MSE")
    logger.info(f"Optimizer: Adam (lr={LEARNING_RATE})")
    logger.info(f"Batch size: {BATCH_SIZE} (optimized for 4GB VRAM)")
    
    # Resume training if requested
    start_epoch = 1
    best_val_loss = float('inf')
    
    if args.resume:
        checkpoint_path = checkpoint_dir / 'best_model.pth'
        if checkpoint_path.exists():
            logger.info("\n" + "=" * 80)
            logger.info("⚠️  POWER FAILURE RECOVERY MODE")
            logger.info("=" * 80)
            
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Load model state
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("✓ Loaded model weights")
            
            # Load optimizer state
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("✓ Loaded optimizer state")
            
            # Load scaler state if using AMP
            if USE_AMP and checkpoint.get('scaler_state_dict'):
                scaler = GradScaler()
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
                logger.info("✓ Loaded AMP scaler state")
            
            # Set starting epoch and best loss
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_val_loss', checkpoint.get('val_loss', float('inf')))
            
            logger.info(f"\n⚠️  Resuming from Epoch {checkpoint['epoch']} with Val Loss {best_val_loss:.4f}")
            logger.info(f"Will continue training from Epoch {start_epoch} to {NUM_EPOCHS}")
            logger.info("=" * 80 + "\n")
        else:
            logger.warning(f"\n⚠️  --resume flag provided but no checkpoint found at {checkpoint_path}")
            logger.warning("Starting training from scratch...\n")
    
    # Create trainer with AMP
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=checkpoint_dir,
        num_epochs=NUM_EPOCHS,
        use_amp=USE_AMP,
        start_epoch=start_epoch
    )
    
    # Set best validation loss if resuming
    if args.resume and best_val_loss != float('inf'):
        trainer.best_val_loss = best_val_loss
    
    # Train
    try:
        trainer.train()
    except RuntimeError as e:
        if "out of memory" in str(e):
            logger.error("=" * 80)
            logger.error("OUT OF MEMORY ERROR!")
            logger.error("=" * 80)
            logger.error("Try reducing BATCH_SIZE further (current: 2)")
            logger.error("Or reduce TARGET_SIZE from 512 to 384")
            logger.error("=" * 80)
        raise


if __name__ == "__main__":
    main()
