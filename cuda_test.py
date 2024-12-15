import torch

# Check CUDA availability
print(f"CUDA is available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    # Number of GPUs
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    # Memory info
    print(f"\nGPU Memory Usage:")
    print(f"Allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f}MB")
    print(f"Cached: {torch.cuda.memory_reserved(0)/1024**2:.2f}MB")
    print(f"PyTorch version: {torch.__version__}")
