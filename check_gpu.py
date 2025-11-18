#!/usr/bin/env python3
"""
ตรวจสอบว่าระบบรองรับ GPU หรือไม่
"""
import torch
import sys

def check_gpu_support():
    print("GPU Support Check")
    print("=" * 50)

    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}: {props.name}")
            print(f"  - Total Memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"  - Compute Capability: {props.major}.{props.minor}")
            print(f"  - Multiprocessors: {props.multi_processor_count}")
        
        print(f"\nCurrent GPU memory:")
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            allocated = torch.cuda.memory_allocated() / 1024**3
            cached = torch.cuda.memory_reserved() / 1024**3
            print(f"  GPU {i}: {allocated:.2f} GB allocated, {cached:.2f} GB cached")
            
        print(f"\nTesting GPU tensor operations...")
        try:
            device = torch.device("cuda:0")
            x = torch.randn(1000, 1000, device=device)
            y = torch.randn(1000, 1000, device=device) 
            z = torch.matmul(x, y)
            print("GPU tensor operations working!")
            print(f"Result tensor shape: {z.shape}")
            print(f"Result tensor device: {z.device}")
        except Exception as e:
            print(f"GPU test failed: {e}")
            
    else:
        print("CUDA not available")
        print("\nPossible solutions:")
        print("1. Install NVIDIA GPU drivers")
        print("2. Install CUDA toolkit") 
        print("3. Install PyTorch with CUDA support:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n💡 Recommended device for training: {device}")
    
    return cuda_available

if __name__ == "__main__":
    check_gpu_support()