# Multi-Architecture VLLM + Axolotl Docker Image

This Docker image combines VLLM (for high-performance LLM inference) and Axolotl (for LLM fine-tuning) in a single multi-architecture image that supports both x86_64 and ARM64 platforms, including NVIDIA GH200 chips.

## Features

- **Multi-Architecture Support**: Works on both x86_64 (AMD64) and ARM64 (aarch64) platforms
- **VLLM v0.10.0**: High-performance inference engine with optimized kernels
- **Axolotl v0.11.0.post1**: Complete LLM fine-tuning framework with DeepSpeed, Flash Attention 2, and more
- **CUDA 12.8.1**: Latest CUDA support for optimal GPU performance
- **Python 3.12**: Modern Python environment
- **Flash Attention 2 v2.8.0.post2**: Built from source for both architectures
- **Xformers v0.0.31.post1**: Built from source for both architectures
- **Bitsandbytes 0.46.1**: Built from source for ARM64, optimized quantization support
- **Optimized for GH200**: Special optimizations for NVIDIA Grace Hopper Superchip

## Architecture-Specific Optimizations

### x86_64 (AMD64)
- MAX_JOBS: 28
- NVCC_THREADS: 2
- TORCH_CUDA_ARCH_LIST: "7.5 8.0 8.9 9.0a+PTX"
- Xformers built from source for optimized performance
- Flash Attention 2 built from source (MAX_JOBS reduced by half for memory efficiency)
- Mamba SSM and Causal Conv1D built from source
- Flashinfer built from source for optimized performance

### ARM64 (aarch64) - Including GH200
- MAX_JOBS: 60 (configurable based on available RAM)
- NVCC_THREADS: 2
- TORCH_CUDA_ARCH_LIST: "7.5 8.0 8.9 9.0a+PTX"
- Xformers built from source for ARM64
- Flash Attention 2 built from source (MAX_JOBS reduced by half for memory efficiency)
- Bitsandbytes built from source with CUDA backend
- Flashinfer built from source for ARM64
- Mamba SSM and Causal Conv1D built from source

## Library Versions

This image includes the following specific versions optimized for multi-architecture deployment:

| Component | Version | Build Method | Architecture Support |
|-----------|---------|--------------|----------------------|
| VLLM | v0.10.0 | Source | x86_64, ARM64 |
| Axolotl | v0.11.0.post1 | Source | x86_64, ARM64 |
| PyTorch | 2.7.1+cu128 | Wheel | x86_64, ARM64 |
| PyTorch Triton | 3.3.0 | Wheel | x86_64, ARM64 |
| TorchVision | 0.22.1 | Wheel | x86_64, ARM64 |
| Flash Attention 2 | v2.8.0.post2 | Source | x86_64, ARM64 |
| Xformers | v0.0.31.post1 | Source | x86_64, ARM64 |
| Bitsandbytes | 0.46.1 | Source (ARM64), Wheel (x86_64) | x86_64, ARM64 |
| Flashinfer | v0.2.9 | Source | x86_64, ARM64 |
| Mamba SSM | main | Source | x86_64, ARM64 |
| Causal Conv1D | main | Source | x86_64, ARM64 |

## Building the Image

### Standard Build
```bash
# Build for current architecture
docker build -t vllm-axolotl:latest -f Dockerfile .

# Build for specific architecture
docker buildx build --platform linux/amd64 -t vllm-axolotl:amd64 -f Dockerfile .
docker buildx build --platform linux/arm64 -t vllm-axolotl:arm64 -f Dockerfile .
```

### Building on GH200 (High RAM Systems)
When building on systems with high RAM (like GH200 with 480GB), you may need to increase Docker's memory limits:

```bash
# Option 1: Remove memory limits (recommended for GH200)
docker build \
  --memory 0 \
  --memory-swap 0 \
  --shm-size 16G \
  -t vllm-axolotl:latest -f Dockerfile .

# Option 2: Using buildx
docker buildx build \
  --builder default \
  --memory 0 --memory-swap 0 --shm-size 16G \
  -t vllm-axolotl:latest -f Dockerfile .
```

### Build Arguments
You can customize the build process with these arguments:

```bash
docker build \
  --build-arg CUDA_VERSION=12.8.1 \
  --build-arg PYTHON_VERSION=3.12 \
  --build-arg AMD64_MAX_JOBS=28 \
  --build-arg ARM64_MAX_JOBS=60 \
  --build-arg VLLM_REF=v0.10.0 \
  --build-arg AXOLOTL_REF=v0.11.0.post1 \
  --build-arg BITSANDBYTES_REF=0.46.1 \
  --build-arg XFORMERS_REF=v0.0.31.post1 \
  --build-arg FLASH_ATTN_REF=v2.8.0.post2 \
  --build-arg FLASHINFER_REF=v0.2.9 \
  -t vllm-axolotl:custom -f Dockerfile .
```

## Usage

### Running Axolotl Training

```bash
# Run training with a configuration file (examples are included in the image)
docker run --gpus all -it \
  -v $(pwd)/axolotl-examples:/workspace/axolotl-examples \
  vllm-axolotl:latest \
  axolotl train /workspace/axolotl/examples/llama-3/lora.yml

# Run with your own config
docker run --gpus all -it \
  -v $(pwd)/configs:/configs \
  vllm-axolotl:latest \
  axolotl train /configs/my_training_config.yml
```

### VLLM Inference Server

```bash
# Start VLLM server (default behavior if no command specified)
docker run --gpus all -p 8000:8000 vllm-axolotl:latest

# Start VLLM with specific model
docker run --gpus all -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm-axolotl:latest \
  vllm serve meta-llama/Llama-2-7b-hf --host 0.0.0.0

# With custom parameters
docker run --gpus all -p 8000:8000 \
  vllm-axolotl:latest \
  vllm serve mistralai/Mistral-7B-v0.1 \
    --host 0.0.0.0 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.95
```

### Interactive Development

```bash
# Start bash shell in workspace
docker run --gpus all -it vllm-axolotl:latest bash

# Start in Axolotl directory
docker run --gpus all -it -w /workspace/axolotl vllm-axolotl:latest bash

# With volume mounts for development
docker run --gpus all -it \
  -v $(pwd):/workspace/dev \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm-axolotl:latest bash
```

### Using Accelerate for Distributed Training

```bash
# Single GPU
docker run --gpus all -it \
  -v $(pwd)/configs:/configs \
  vllm-axolotl:latest \
  accelerate launch -m axolotl.cli.train /configs/config.yml

# Multi-GPU with DeepSpeed
docker run --gpus all -it \
  -v $(pwd)/configs:/configs \
  vllm-axolotl:latest \
  accelerate launch --multi_gpu --num_processes 8 \
    -m axolotl.cli.train /configs/config.yml
```

## Environment Variables

- `HF_HUB_ENABLE_HF_TRANSFER=1`: Enables fast HuggingFace downloads
- `TORCH_CUDA_ARCH_LIST`: GPU architectures (default: "7.5 8.0 8.9 9.0 10.0+PTX")
- `CUDA_HOME=/usr/local/cuda`: CUDA installation path
- `VIRTUAL_ENV=/workspace/.venv`: Python virtual environment path

### Disabling Flash Attention (if needed)
```bash
docker run --gpus all -it \
  -e FLASH_ATTENTION_FORCE_DISABLE=1 \
  vllm-axolotl:latest \
  axolotl train config.yml
```

## Included Components

### Core Frameworks
- **PyTorch 2.7.1+cu128**: Latest PyTorch with CUDA 12.8 support
- **PyTorch Triton 3.3.0**: Accelerated GPU kernel library
- **TorchVision 0.22.1**: Computer vision utilities
- **VLLM v0.10.0**: High-performance LLM inference
- **Axolotl v0.11.0.post1**: Complete fine-tuning framework
- **Flash Attention 2 v2.8.0.post2**: Memory-efficient attention mechanism
- **Xformers v0.0.31.post1**: Efficient transformer implementations
- **Flashinfer v0.2.9**: High-performance attention kernels

### Training Components
- DeepSpeed for distributed training
- Bitsandbytes 0.46.1 for quantization (built from source on ARM64)
- PEFT/LoRA support
- Unsloth optimizations
- CutCrossEntropy
- Mamba SSM support (built from source for both architectures)
- Causal Conv1D (built from source for both architectures)
- HuggingFace Accelerate

### Infrastructure Tools
- AWS CLI for S3 access
- Git LFS for large model files
- S3FS for mounting S3 buckets
- HuggingFace Accelerate
- Weights & Biases support

## Troubleshooting

### Out of Memory During Build
If you encounter OOM errors during compilation (especially Flash Attention, Xformers, or Bitsandbytes):
1. Use the `--memory 0` Docker build flag to remove memory limits
2. The Dockerfile already reduces parallelism for Flash Attention (MAX_JOBS/2)
3. Ensure sufficient swap space is available
4. Consider reducing ARM64_MAX_JOBS or AMD64_MAX_JOBS build arguments if needed

### Flash Attention Import Error
If you get "ImportError: Flash Attention 2 is not available":
- The package is installed but may not be compatible with your GPU
- Try setting `FLASH_ATTENTION_FORCE_DISABLE=1`
- Verify your GPU compute capability is >= 7.5

### Performance Optimization
For best performance on GH200:
- Use the full GPU memory: `--gpu-memory-utilization 0.95`
- Enable tensor parallelism for large models
- Use appropriate batch sizes for your workload

## Contributing

Contributions are welcome! Please submit pull requests or issues to improve multi-architecture support, add new features, or fix bugs.

## License

This Docker image combines multiple open-source projects. Please refer to individual project licenses:
- VLLM: Apache 2.0
- Axolotl: Apache 2.0
- Flash Attention: BSD 3-Clause
- PyTorch: BSD 3-Clause

# axolotl-multi-arch

