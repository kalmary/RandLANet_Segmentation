#!/usr/bin/env python3
"""
Benchmark script to test RandLANet memory optimizations
"""

import torch
import time
import sys
import pathlib as pth

# Add src to path
src_dir = pth.Path(__file__).parent
sys.path.append(str(src_dir))

from model_pipeline.RandLANet_CB import RandLANet
from utils.memory_utils import get_memory_info, MemoryMonitor, cleanup_memory

def benchmark_model(batch_size=2, num_points=4096, num_classes=10, memory_efficient=True):
    """Benchmark the optimized RandLANet model"""
    
    model_config = {
        'd_in': 4,
        'num_neighbors': 16,
        'decimation': 2,
        'encoder_layers': [
            {'d_in': 8, 'd_out': 32},
            {'d_in': 64, 'd_out': 64},
            {'d_in': 128, 'd_out': 128},
        ],
        'decoder_layers': [
            {'d_in': 384, 'd_out': 128},
            {'d_in': 192, 'd_out': 64},
            {'d_in': 96, 'd_out': 8}
        ],
        'fc_start': {'d_out': 8},
        'fc_end': {'layers': [32, 16], 'dropout': 0.3},
        'max_voxel_dim': 20.
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = RandLANet(model_config=model_config, num_classes=num_classes).to(device)
    
    if memory_efficient:
        model.enable_memory_efficient_mode()
        print("Memory efficient mode enabled")
    
    # Print model memory usage
    model_memory = model.get_memory_usage()
    print(f"Model memory usage: {model_memory['total_mb']:.2f} MB")
    
    # Create test input
    dummy_input = torch.randn(batch_size, num_points, model_config['d_in']).to(device)
    
    print(f"\nBenchmarking with batch_size={batch_size}, num_points={num_points}")
    print("=" * 60)
    
    # Warmup
    with torch.no_grad():
        _ = model(dummy_input, memory_efficient=memory_efficient)
    
    cleanup_memory()
    
    # Benchmark
    start_memory = get_memory_info()
    print(f"Initial GPU memory: {start_memory.get('gpu_memory_allocated_mb', 0):.1f} MB")
    
    times = []
    peak_memory = 0
    
    for i in range(5):
        cleanup_memory()
        
        with MemoryMonitor(f"Forward pass {i+1}", verbose=True):
            start_time = time.time()
            
            with torch.no_grad():
                output = model(dummy_input, memory_efficient=memory_efficient)
            
            end_time = time.time()
            times.append(end_time - start_time)
            
            current_memory = get_memory_info()
            current_gpu_mem = current_memory.get('gpu_memory_allocated_mb', 0)
            peak_memory = max(peak_memory, current_gpu_mem)
    
    # Results
    avg_time = sum(times) / len(times)
    print(f"\nResults:")
    print(f"Average forward pass time: {avg_time:.3f} seconds")
    print(f"Peak GPU memory usage: {peak_memory:.1f} MB")
    print(f"Output shape: {output.shape}")
    
    # Verify output
    expected_shape = (batch_size, num_classes, num_points)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    print("✓ Output shape verification passed")
    
    return avg_time, peak_memory

if __name__ == "__main__":
    print("RandLANet Memory Optimization Benchmark")
    print("=" * 50)
    
    # Test different configurations
    configs = [
        (1, 2048, "Small"),
        (2, 4096, "Medium"),
        (1, 8192, "Large"),
    ]
    
    for batch_size, num_points, size_name in configs:
        print(f"\n{size_name} configuration:")
        try:
            avg_time, peak_memory = benchmark_model(
                batch_size=batch_size, 
                num_points=num_points,
                memory_efficient=True
            )
            print(f"✓ {size_name} test completed successfully")
        except Exception as e:
            print(f"✗ {size_name} test failed: {e}")
    
    print("\nBenchmark completed!")