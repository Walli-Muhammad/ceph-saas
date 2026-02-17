"""
Quick script to check PyTorch and CUDA setup
"""
import torch

print("=" * 80)
print("PYTORCH & CUDA CHECK")
print("=" * 80)
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Count: {torch.cuda.device_count()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("\n✓ CUDA is ready! You can start training.")
else:
    print("\n⚠️  CUDA is NOT available!")
    print("\nPossible reasons:")
    print("1. PyTorch installed without CUDA support (CPU-only version)")
    print("2. NVIDIA drivers not installed")
    print("3. No NVIDIA GPU in system")
    print("\nTo install PyTorch with CUDA support:")
    print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

print("=" * 80)
