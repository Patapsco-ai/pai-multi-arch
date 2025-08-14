# Builds a multi-arch VLLM and Axolotl Docker image with support for both x86_64 and ARM64 architectures.
# There are 2 workspaces:
# 1. /workspace/vllm: Contains the VLLM code and dependencies
# 2. /workspace/axolotl: Contains the Axolotl code and dependencies

# Configure MAX_JOBS and NVCC_THREADS based on architecture. Changes will effect memory usage and build time.
ARG AMD64_MAX_JOBS=28
ARG ARM64_MAX_JOBS=60
ARG AMD64_NVCC_THREADS=2
ARG ARM64_NVCC_THREADS=2

# Global build arguments
# These are used to set the CUDA version, image distribution, Python version, and Torch CUDA architecture list.
ARG CUDA_VERSION=12.8.1

# Must match the CUDA version above
ARG CUDA_VERSION_SHORT=cu128
ARG IMAGE_DISTRO=ubuntu24.04
ARG PYTHON_VERSION=3.12
ARG TORCH_CUDA_ARCH_LIST="7.5 8.0 8.9 9.0 9.0a+PTX"
# BitsandBytes requires different format of CUDA arch list
ARG CUDA_CAPABILITIES="75;80;86;87;89;90"

# Main App Versions
ARG VLLM_REF=v0.10.0
ARG VLLM_PATCHES=""
ARG AXOLOTL_REF=v0.12.1
ARG AXOLOTL_PATCHES=""

# Library Versions
ARG PYTORCH_REF=2.7.1
ARG TORCHVISION_REF=0.22.1
ARG PYTORCH_TRITON_REF=3.3.0
ARG BITSANDBYTES_REF=0.46.1
ARG XFORMERS_REF=v0.0.31
ARG FLASH_ATTN_REF=v2.8.0.post2
# Using RC1 version due to: https://github.com/flashinfer-ai/flashinfer/issues/1256
ARG FLASHINFER_REF=v0.2.8rc1
ARG MAMBA_SSM_REF=main
ARG CAUSAL_CONV1D_REF=main


# ---------- Builder Base ----------
FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-devel-${IMAGE_DISTRO} AS base

# Set arch lists for all targets
# 'a' suffix is not forward compatible but enables all optimizations
ARG TORCH_CUDA_ARCH_LIST
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ARG VLLM_FA_CMAKE_GPU_ARCHES="80-real;90-real"
ENV VLLM_FA_CMAKE_GPU_ARCHES=${VLLM_FA_CMAKE_GPU_ARCHES}
ARG CUDA_VERSION_SHORT
ARG PYTORCH_REF
ARG TORCHVISION_REF
ARG PYTORCH_TRITON_REF
# Update apt packages and install dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install -y --no-install-recommends --allow-change-held-packages \
    curl \
    gcc-13 g++-13 \
    git \
    libibverbs-dev \
    libjpeg-turbo8-dev \
    libnccl-dev=2.27.7-1+cuda12.9 \
    libnccl2=2.27.7-1+cuda12.9 \
    libpng-dev \
    zlib1g-dev \
    vim \
    wget \
    rsync \
    s3fs \
    libaio-dev \
    pkg-config \
    less \
    tmux && \
    rm -rf /var/lib/apt/lists/*

# Set compiler paths
ENV CC=/usr/bin/gcc-13
ENV CXX=/usr/bin/g++-13

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

# Setup build vllm-workspace
WORKDIR /workspace

# Prep build venv
ARG PYTHON_VERSION
RUN uv venv -p ${PYTHON_VERSION} --seed --python-preference only-managed
ENV VIRTUAL_ENV=/workspace/.venv
ENV PATH=${VIRTUAL_ENV}/bin:${PATH}
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

RUN uv pip install --index-url https://download.pytorch.org/whl/cu128 torch==${PYTORCH_REF}+${CUDA_VERSION_SHORT} torchvision==${TORCHVISION_REF} torchaudio==${PYTORCH_REF} pytorch_triton==${PYTORCH_TRITON_REF};

FROM base AS build-base
# Pass build arguments to this stage
ARG AMD64_MAX_JOBS
ARG ARM64_MAX_JOBS
ARG AMD64_NVCC_THREADS
ARG ARM64_NVCC_THREADS
ARG TORCH_CUDA_ARCH_LIST
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

RUN apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /wheels
# Install build deps that aren't in project requirements files
# Make sure to upgrade setuptools to avoid triton build bug
# cmake '4.x' isn't parsed right by some tools yet
RUN uv pip install -U build "cmake<4" ninja pybind11 "setuptools<=76" wheel

# Build xformers only for aarch64 architecture
FROM build-base AS build-libs-aarch64
# Pass build arguments to this stage
ARG AMD64_MAX_JOBS
ARG ARM64_MAX_JOBS
ARG AMD64_NVCC_THREADS
ARG ARM64_NVCC_THREADS
ARG TORCH_CUDA_ARCH_LIST
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ARG CUDA_CAPABILITIES

# Build xformers
ARG XFORMERS_REF
ARG XFORMERS_BUILD_VERSION=${XFORMERS_REF}+cu128
ENV BUILD_VERSION=${XFORMERS_BUILD_VERSION:-${XFORMERS_REF#v}}

RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "aarch64" ]; then \
        MAX_JOBS=${ARM64_MAX_JOBS} NVCC_THREADS=${ARM64_NVCC_THREADS} && \
        echo "Building xformers with aarch64 parameters: MAX_JOBS=${ARM64_MAX_JOBS}, NVCC_THREADS=${ARM64_NVCC_THREADS}" && \
        git clone https://github.com/facebookresearch/xformers.git && \
        cd xformers && \
        git checkout ${XFORMERS_REF} && \
        git submodule sync --recursive && \
        git submodule update --init --recursive -j 4 && \
        TORCH_CUDA_ARCH_LIST="7.5 8.0+PTX 9.0a" MAX_JOBS=${ARM64_MAX_JOBS} NVCC_THREADS=${ARM64_NVCC_THREADS} uv build --wheel --no-build-isolation --quiet -o /wheels; \
    elif [ "$ARCH" = "amd64" ] || [ "$ARCH" = "x86_64" ]; then \
        MAX_JOBS=${AMD64_MAX_JOBS} NVCC_THREADS=${AMD64_NVCC_THREADS} && \
        echo "Building xformers with amd64/x86_64 parameters: MAX_JOBS=${AMD64_MAX_JOBS}, NVCC_THREADS=${AMD64_NVCC_THREADS}" && \
        git clone https://github.com/facebookresearch/xformers.git && \
        cd xformers && \
        git checkout ${XFORMERS_REF} && \
        git submodule sync --recursive && \
        git submodule update --init --recursive -j 4 && \
        TORCH_CUDA_ARCH_LIST="7.5 8.0+PTX 9.0a" MAX_JOBS=${AMD64_MAX_JOBS} NVCC_THREADS=${AMD64_NVCC_THREADS} uv build --wheel --no-build-isolation --quiet -o /wheels; \
    fi

# Build flashinfer
ARG FLASHINFER_REF
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "aarch64" ]; then \
        uv pip install "packaging>=24.2" "setuptools>=79.0" && \
        MAX_JOBS=${ARM64_MAX_JOBS} NVCC_THREADS=${ARM64_NVCC_THREADS} && \
        echo "Building flashinfer with aarch64 parameters: MAX_JOBS=${ARM64_MAX_JOBS}, NVCC_THREADS=${ARM64_NVCC_THREADS}" && \
        git clone https://github.com/flashinfer-ai/flashinfer.git && \
        cd flashinfer && \
        git checkout ${FLASHINFER_REF} && \
        git submodule sync --recursive && \
        git submodule update --init --recursive && \
        MAX_JOBS=$((${ARM64_MAX_JOBS}/2)) NVCC_THREADS=${ARM64_NVCC_THREADS} python -m flashinfer.aot && \
        ln -s ../aot-ops build/aot-ops-package-dir && \
        MAX_JOBS=$((${ARM64_MAX_JOBS}/2)) NVCC_THREADS=${ARM64_NVCC_THREADS} uv build --wheel --no-build-isolation --quiet -o /wheels; \
    elif [ "$ARCH" = "amd64" ] || [ "$ARCH" = "x86_64" ]; then \
        uv pip install "packaging>=24.2" "setuptools>=79.0" && \
        MAX_JOBS=${AMD64_MAX_JOBS} NVCC_THREADS=${AMD64_NVCC_THREADS} && \
        echo "Building flashinfer with amd64/x86_64 parameters: MAX_JOBS=${AMD64_MAX_JOBS}, NVCC_THREADS=${AMD64_NVCC_THREADS}" && \
        git clone https://github.com/flashinfer-ai/flashinfer.git && \
        cd flashinfer && \
        git checkout ${FLASHINFER_REF} && \
        git submodule sync --recursive && \
        git submodule update --init --recursive && \
        MAX_JOBS=$((${AMD64_MAX_JOBS}/2)) NVCC_THREADS=${AMD64_NVCC_THREADS} python -m flashinfer.aot && \
        ln -s ../aot-ops build/aot-ops-package-dir && \
        MAX_JOBS=$((${AMD64_MAX_JOBS}/2)) NVCC_THREADS=${AMD64_NVCC_THREADS} uv build --wheel --no-build-isolation --quiet -o /wheels; \
    fi

# Build bitsandbytes
ARG BITSANDBYTES_REF
ARG BITSANDBYTES_BUILD_VERSION=${BITSANDBYTES_REF}+cu128
ENV BUILD_VERSION=${BITSANDBYTES_BUILD_VERSION:-${BITSANDBYTES_REF#v}}
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "aarch64" ]; then \
        MAX_JOBS=${ARM64_MAX_JOBS} NVCC_THREADS=${ARM64_NVCC_THREADS} && \
        echo "Building bitsandbytes with aarch64 parameters: MAX_JOBS=${ARM64_MAX_JOBS}, NVCC_THREADS=${ARM64_NVCC_THREADS}" && \
        git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git && \
        cd bitsandbytes && \
        git checkout ${BITSANDBYTES_REF} && \
        cmake -DCOMPUTE_BACKEND=cuda -DCOMPUTE_CAPABILITY=${CUDA_CAPABILITIES} -S . && make -j $((${ARM64_MAX_JOBS}/2)) && \
        uv pip install scikit_build_core && \
        MAX_JOBS=${ARM64_MAX_JOBS} NVCC_THREADS=${ARM64_NVCC_THREADS} uv build --wheel --no-build-isolation --quiet -o /wheels; \
    fi

# Build flash-attention
ARG FLASH_ATTN_REF
ARG FLASH_ATTN_BUILD_VERSION=${FLASH_ATTN_REF}+cu128
ENV BUILD_VERSION=${FLASH_ATTN_BUILD_VERSION:-${FLASH_ATTN_REF#v}}
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "aarch64" ]; then \
        MAX_JOBS=$((${ARM64_MAX_JOBS}/2)) NVCC_THREADS=${ARM64_NVCC_THREADS} && \
        echo "Building flash-attn with aarch64 parameters: MAX_JOBS=$((${ARM64_MAX_JOBS}/2)), NVCC_THREADS=${ARM64_NVCC_THREADS}" && \
        git clone -b ${FLASH_ATTN_REF} --recurse https://github.com/Dao-AILab/flash-attention.git && \
        cd flash-attention && \
        MAX_JOBS=$((${ARM64_MAX_JOBS}/2)) NVCC_THREADS=${ARM64_NVCC_THREADS} uv build --wheel --no-build-isolation --quiet -o /wheels; \
    elif [ "$ARCH" = "amd64" ] || [ "$ARCH" = "x86_64" ]; then \
        MAX_JOBS=$((${AMD64_MAX_JOBS}/2)) NVCC_THREADS=${AMD64_NVCC_THREADS} && \
        echo "Building flash-attn with amd64/x86_64 parameters: MAX_JOBS=$((${AMD64_MAX_JOBS}/2)), NVCC_THREADS=${AMD64_NVCC_THREADS}" && \
        git clone -b ${FLASH_ATTN_REF} --recurse https://github.com/Dao-AILab/flash-attention.git && \
        cd flash-attention && \
        MAX_JOBS=$((${AMD64_MAX_JOBS}/2)) NVCC_THREADS=${AMD64_NVCC_THREADS} uv build --wheel --no-build-isolation --quiet -o /wheels; \
    fi

# Build causal-conv1d
ARG CAUSAL_CONV1D_REF
ARG CAUSAL_CONV1D_BUILD_VERSION=${CAUSAL_CONV1D_REF}+cu128
ENV BUILD_VERSION=${CAUSAL_CONV1D_BUILD_VERSION:-${CAUSAL_CONV1D_REF#v}}
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "aarch64" ]; then \
        MAX_JOBS=${ARM64_MAX_JOBS} NVCC_THREADS=${ARM64_NVCC_THREADS} && \
        echo "Building causal-conv1d with aarch64 parameters: MAX_JOBS=${ARM64_MAX_JOBS}, NVCC_THREADS=${ARM64_NVCC_THREADS}" && \
        git clone https://github.com/Dao-AILab/causal-conv1d.git && \
        cd causal-conv1d && \
        if [ -n "${CAUSAL_CONV1D_REF}" ]; then git checkout ${CAUSAL_CONV1D_REF}; else git checkout main; fi && \
        MAX_JOBS=${ARM64_MAX_JOBS} NVCC_THREADS=${ARM64_NVCC_THREADS} uv build --wheel --no-build-isolation --quiet -o /wheels; \
    elif [ "$ARCH" = "amd64" ] || [ "$ARCH" = "x86_64" ]; then \
        MAX_JOBS=${AMD64_MAX_JOBS} NVCC_THREADS=${AMD64_NVCC_THREADS} && \
        echo "Building causal-conv1d with amd64/x86_64 parameters: MAX_JOBS=${AMD64_MAX_JOBS}, NVCC_THREADS=${AMD64_NVCC_THREADS}" && \
        git clone https://github.com/Dao-AILab/causal-conv1d.git && \
        cd causal-conv1d && \
        if [ -n "${CAUSAL_CONV1D_REF}" ]; then git checkout ${CAUSAL_CONV1D_REF}; else git checkout main; fi && \
        MAX_JOBS=${AMD64_MAX_JOBS} NVCC_THREADS=${AMD64_NVCC_THREADS} uv build --wheel --no-build-isolation --quiet -o /wheels; \
    fi

# Build mamba-ssm
ARG MAMBA_SSM_REF
ARG MAMBA_SSM_BUILD_VERSION=${MAMBA_SSM_REF}+cu128
ENV BUILD_VERSION=${MAMBA_SSM_BUILD_VERSION:-${MAMBA_SSM_REF#v}}
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "aarch64" ]; then \
        MAX_JOBS=${ARM64_MAX_JOBS} NVCC_THREADS=${ARM64_NVCC_THREADS} && \
        echo "Building mamba-ssm with aarch64 parameters: MAX_JOBS=${ARM64_MAX_JOBS}, NVCC_THREADS=${ARM64_NVCC_THREADS}" && \
        git clone https://github.com/state-spaces/mamba.git && \
        cd mamba && \
        if [ -n "${MAMBA_SSM_REF}" ]; then git checkout ${MAMBA_SSM_REF}; else git checkout main; fi && \
        MAX_JOBS=${ARM64_MAX_JOBS} NVCC_THREADS=${ARM64_NVCC_THREADS} uv build --wheel --no-build-isolation --quiet -o /wheels; \
    elif [ "$ARCH" = "amd64" ] || [ "$ARCH" = "x86_64" ]; then \
        MAX_JOBS=${AMD64_MAX_JOBS} NVCC_THREADS=${AMD64_NVCC_THREADS} && \
        echo "Building mamba-ssm with amd64/x86_64 parameters: MAX_JOBS=${AMD64_MAX_JOBS}, NVCC_THREADS=${AMD64_NVCC_THREADS}" && \
        git clone https://github.com/state-spaces/mamba.git && \
        cd mamba && \
        if [ -n "${MAMBA_SSM_REF}" ]; then git checkout ${MAMBA_SSM_REF}; else git checkout main; fi && \
        MAX_JOBS=${AMD64_MAX_JOBS} NVCC_THREADS=${AMD64_NVCC_THREADS} uv build --wheel --no-build-isolation --quiet -o /wheels; \
    fi


FROM build-base AS build-vllm
# Pass build arguments to this stage
ARG AMD64_MAX_JOBS
ARG ARM64_MAX_JOBS
ARG AMD64_NVCC_THREADS
ARG ARM64_NVCC_THREADS
ARG TORCH_CUDA_ARCH_LIST
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

COPY --from=build-libs-aarch64 /wheels/ /wheels/

RUN find /wheels/ -name "*.whl" -exec uv pip install {} + || echo "No wheels to install"

ARG VLLM_REF
ARG VLLM_PATCHES
ARG VLLM_BUILD_VERSION
ENV BUILD_VERSION=${VLLM_BUILD_VERSION:-${VLLM_REF#v}}
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${BUILD_VERSION:-:}
RUN git clone https://github.com/vllm-project/vllm.git

RUN cd vllm && \
    git checkout ${VLLM_REF} && \
    # Apply patches if provided (kept for future use)
    if [ -n "${VLLM_PATCHES}" ] && [ "${VLLM_PATCHES}" != "" ]; then \
        echo "Applying patches: ${VLLM_PATCHES}"; \
        for patch in ${VLLM_PATCHES}; do \
            echo "Applying patch: $patch"; \
            curl -L "https://github.com/vllm-project/vllm/commit/$patch.patch" | git apply || { \
                echo "Failed to apply patch $patch"; \
                exit 1; \
            }; \
        done; \
    else \
        echo "No patches to apply"; \
    fi && \
    python use_existing_torch.py && \
    uv pip install -r requirements/build.txt && \
    ARCH=$(uname -m) && \
    if [ "$ARCH" = "aarch64" ]; then \
        MAX_JOBS=${ARM64_MAX_JOBS} NVCC_THREADS=${ARM64_NVCC_THREADS} uv build --wheel --no-build-isolation --quiet -o /wheels; \
    else \
        MAX_JOBS=${AMD64_MAX_JOBS} NVCC_THREADS=${AMD64_NVCC_THREADS} uv build --wheel --no-build-isolation --quiet -o /wheels; \
    fi

FROM build-base AS build-axolotl
# Pass build arguments to this stage
ARG AMD64_MAX_JOBS
ARG ARM64_MAX_JOBS
ARG AMD64_NVCC_THREADS
ARG ARM64_NVCC_THREADS
ARG TORCH_CUDA_ARCH_LIST
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

COPY --from=build-libs-aarch64 /wheels/ /wheels/

RUN find /wheels/ -name "*.whl" -exec uv pip install {} + || echo "No wheels to install"

ARG AXOLOTL_REF
ARG AXOLOTL_PATCHES
ARG AXOLOTL_BUILD_VERSION=${AXOLOTL_REF}+cu128
ENV BUILD_VERSION=${AXOLOTL_BUILD_VERSION:-${AXOLOTL_REF#v}}
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${BUILD_VERSION:-:}
RUN git clone https://github.com/axolotl-ai-cloud/axolotl.git


RUN cd axolotl && \
    git checkout ${AXOLOTL_REF} && \
    # Apply patches if provided (kept for future use)
    if [ -n "${AXOLOTL_PATCHES}" ] && [ "${AXOLOTL_PATCHES}" != "" ]; then \
        echo "Applying patches: ${AXOLOTL_PATCHES}"; \
        for patch in ${AXOLOTL_PATCHES}; do \
            echo "Applying patch: $patch"; \
            curl -L "https://github.com/axolotl-ai-cloud/axolotl/commit/$patch.patch" | git apply || { \
                echo "Failed to apply patch $patch"; \
                exit 1; \
            }; \
        done; \
    else \
        echo "No patches to apply"; \
    fi && \
    # Modify requirements to be more flexible with bitsandbytes version
    if grep -q "bitsandbytes==" requirements.txt 2>/dev/null; then \
        sed -i 's/bitsandbytes==[0-9.]*/bitsandbytes>=0.46.1/g' requirements.txt || true; \
    fi && \
    if grep -q "mamba-ssm==" requirements.txt 2>/dev/null; then \
            sed -i 's/mamba-ssm==[0-9.]*/mamba-ssm>=1.2.0/g' requirements.txt || true; \
    fi && \
    if grep -q "mistral-common==" requirements.txt 2>/dev/null; then \
            sed -i 's/mistral-common==[0-9.]*/mistral-common>=1.8.3/g' requirements.txt || true; \
    fi && \
    uv pip install -r requirements.txt && \
    ARCH=$(uname -m) && \
    if [ "$ARCH" = "aarch64" ]; then \
        MAX_JOBS=${ARM64_MAX_JOBS} NVCC_THREADS=${ARM64_NVCC_THREADS} uv build --wheel --no-build-isolation --quiet -o /wheels; \
    else \
        MAX_JOBS=${AMD64_MAX_JOBS} NVCC_THREADS=${AMD64_NVCC_THREADS} uv build --wheel --no-build-isolation --quiet -o /wheels; \
    fi

FROM base AS patapsco-llm
ARG TORCH_CUDA_ARCH_LIST
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ARG CUDA_VERSION_SHORT
ARG PYTORCH_REF
ARG TORCHVISION_REF
ARG PYTORCH_TRITON_REF

RUN mkdir -p /wheels && \
    mkdir -p /workspace/axolotl/examples && \
    mkdir -p /workspace/axolotl/scripts && \
    mkdir -p /workspace/axolotl/deepspeed_configs && \
    mkdir -p /workspace/vllm && \
    ln -s /workspace/vllm /vllm-workspace

# Copy over all of the wheels from previous build stages
COPY --from=build-vllm /wheels/*.whl /wheels/
COPY --from=build-axolotl /wheels/*.whl /wheels/

#Copy Axolotl examples and deepspeed_configs directories
COPY --from=build-axolotl /workspace/axolotl/examples /workspace/axolotl/examples/
COPY --from=build-axolotl /workspace/axolotl/scripts /workspace/axolotl/scripts/
COPY --from=build-axolotl /workspace/axolotl/deepspeed_configs /workspace/axolotl/deepspeed_configs/

# Copy vllm examples directory
COPY --from=build-vllm /workspace/vllm/examples /workspace/vllm/examples/

# Common Packages for VLLM and Axolotl
RUN uv pip install pynvml
# Add additional packages for vLLM OpenAI
RUN uv pip install accelerate hf_transfer modelscope timm boto3 runai-model-streamer runai-model-streamer[s3] tensorizer pyarrow pandas

# Add additional packages for Axolotl
RUN uv pip install awscli transformers-stream-generator optimum scipy py-cpuinfo hjson ninja psutil deepspeed "pydantic>=2.0,<3.0" pytest tensorboardx

# Install and cleanup wheels
RUN find /wheels/ -name "*.whl" -exec uv pip install {} + || echo "No wheels to install"
RUN rm -rf /wheels

# Install Git LFS for handling large model files
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get install -y git-lfs && \
    git lfs install --skip-repo && \
    rm -rf /var/lib/apt/lists/*

# Run Axolotl post-installation scripts
RUN python3 /workspace/axolotl/scripts/unsloth_install.py | sh || echo "Unsloth installation skipped or failed"
RUN python3 /workspace/axolotl/scripts/cutcrossentropy_install.py | sh || echo "CutCrossEntropy installation skipped or failed"

# Clean uv cache
RUN uv clean

# Clean apt cache
RUN apt autoremove --purge -y
RUN apt clean
RUN rm -rf /var/lib/apt/lists/*
RUN rm -rf /var/cache/apt/archives

# Enable hf-transfer
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Create entrypoint script that can run both VLLM and Axolotl
RUN cat > /entrypoint.sh << 'EOF'
#!/bin/bash
set -e

# Default to VLLM if no command specified
if [ $# -eq 0 ]; then
    echo "Starting VLLM server..."
    exec vllm serve
fi

# Check first argument
case "$1" in
    vllm)
        shift
        echo "Starting VLLM..."
        cd /workspace/vllm
        exec vllm "$@"
        ;;
    axolotl)
        shift
        echo "Starting Axolotl..."
        cd /workspace/axolotl
        if [ $# -eq 0 ]; then
            # If no arguments, show help
            exec python -m axolotl.cli.train --help
        else
            exec python -m axolotl.cli.train "$@"
        fi
        ;;
    accelerate)
        shift
        echo "Starting with Accelerate..."
        cd /workspace/axolotl
        exec accelerate launch "$@"
        ;;
    bash|sh)
        exec "$@"
        ;;
    *)
        # If not a recognized command, pass through
        exec "$@"
        ;;
esac
EOF

RUN chmod +x /entrypoint.sh

# Set the flexible entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Default command (can be overridden)
CMD []