import torch
import sys

print("="*40)
print("CUDA VERIFICATION")
print("="*40)
try:
    is_available = torch.cuda.is_available()
    print(f"CUDA Available: {is_available}")
    
    if is_available:
        device_count = torch.cuda.device_count()
        print(f"Device Count: {device_count}")
        for i in range(device_count):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        
        current_device = torch.cuda.current_device()
        print(f"Current Device Index: {current_device}")
    else:
        print("❌ CUDA IS NOT AVAILABLE. Training will run on CPU (Slow).")
        
except Exception as e:
    print(f"Error checking CUDA: {e}")
print("="*40)
