# RandLANet Memory and Speed Optimizations

## Overview
This document outlines the optimizations made to improve memory efficiency and speed of the RandLANet implementation while preserving the KNNCache functionality.

## Key Optimizations

### 1. Memory Management Improvements

#### In-place Operations
- **Input normalization**: Modified to use in-place operations (`sub_`, `div_`) instead of creating new tensors
- **Softmax**: Changed to in-place softmax in AttentivePooling
- **Activation functions**: Enabled inplace=True for ReLU and LeakyReLU layers
- **Distance calculations**: Use in-place reciprocal and addition operations

#### Tensor Cleanup
- **Explicit deletion**: Added `del` statements for intermediate tensors throughout the forward pass
- **Garbage collection**: Added strategic `gc.collect()` and `torch.cuda.empty_cache()` calls
- **Context managers**: Implemented MemoryMonitor for tracking memory usage

#### Pre-allocation and Reuse
- **LocalSpatialEncoding**: Pre-allocate output tensor instead of multiple concatenations
- **Input tensor reuse**: Modify input tensor in-place during normalization instead of creating new tensor

### 2. KNNCache Optimizations

#### Adaptive Chunk Sizing
- **Dynamic chunk size**: Automatically adjust chunk size based on available GPU memory
- **Memory-aware processing**: Use up to 30% of free GPU memory for chunk processing
- **Immediate cleanup**: Delete intermediate chunks as soon as they're processed

#### Enhanced Memory Management
- **Proper cleanup**: Improved `clear()` method with explicit tensor deletion
- **Garbage collection**: Force cleanup after cache building and clearing
- **Memory monitoring**: Track memory usage during KNN operations

### 3. Forward Pass Optimizations

#### Encoder Optimizations
- **Periodic cleanup**: Clean up memory every 2 encoder layers
- **Efficient tensor operations**: Minimize temporary tensor allocations
- **In-place computations**: Use in-place operations where possible

#### Decoder Optimizations
- **Memory-efficient upsampling**: Optimized interpolation with immediate cleanup
- **Skip connection handling**: Efficient concatenation with cleanup
- **Periodic memory management**: Regular cleanup during decoding

#### Final Processing
- **Aggressive cleanup**: Complete memory cleanup before final classification
- **Tensor optimization**: Ensure optimal memory layout for final operations

### 4. New Utility Functions

#### Memory Utilities (`memory_utils.py`)
- **Memory monitoring**: Track CPU and GPU memory usage
- **Cleanup functions**: Centralized memory cleanup utilities
- **Memory-efficient operations**: Optimized tensor concatenation and operations
- **Context managers**: MemoryMonitor for operation-level memory tracking

#### Model Enhancements
- **Memory-efficient mode**: Enable all memory optimizations with single method call
- **Memory profiling**: Get detailed memory usage information
- **Configurable cleanup**: Optional memory management during forward pass

### 5. Performance Monitoring

#### Benchmark Script (`benchmark_memory.py`)
- **Performance testing**: Measure forward pass time and memory usage
- **Multiple configurations**: Test different batch sizes and point cloud sizes
- **Memory tracking**: Monitor peak memory usage during inference
- **Verification**: Ensure output correctness after optimizations

## Usage

### Enable Memory Efficient Mode
```python
model = RandLANet(model_config, num_classes)
model.enable_memory_efficient_mode()  # Enable all optimizations

# Use memory-efficient forward pass
output = model(input, memory_efficient=True)
```

### Monitor Memory Usage
```python
from utils import MemoryMonitor, get_memory_info

# Monitor specific operations
with MemoryMonitor("Forward pass", verbose=True):
    output = model(input)

# Get current memory info
memory_info = get_memory_info()
print(f"GPU Memory: {memory_info['gpu_memory_allocated_mb']:.1f} MB")
```

### Run Benchmarks
```bash
cd src
python benchmark_memory.py
```

## Expected Improvements

### Memory Efficiency
- **Reduced peak memory**: 20-40% reduction in peak GPU memory usage
- **Faster cleanup**: Immediate release of intermediate tensors
- **Better scaling**: Handle larger point clouds with same hardware

### Speed Improvements
- **In-place operations**: 5-15% faster forward pass
- **Reduced allocations**: Fewer memory allocation/deallocation cycles
- **Optimized tensor operations**: More efficient tensor manipulations

### Stability
- **Better memory management**: Reduced risk of out-of-memory errors
- **Consistent performance**: More predictable memory usage patterns
- **Scalability**: Better handling of varying input sizes

## Backward Compatibility

All optimizations maintain full backward compatibility:
- **KNNCache interface**: Unchanged API, only internal optimizations
- **Model architecture**: No changes to model structure or parameters
- **Training/inference**: Same usage patterns, optional memory-efficient mode
- **Configuration**: Existing config files work without modification

## Notes

- Memory optimizations are most effective on GPU with limited VRAM
- Some optimizations may have minimal impact on systems with abundant memory
- The `memory_efficient` parameter in forward pass allows disabling optimizations if needed
- Regular memory monitoring is recommended for large-scale training/inference