#!/bin/bash
# Mini-Kernel Agent Setup Script
# This script sets up the environment using Docker

set -e

echo "=============================================="
echo "  Mini-Kernel Agent - Setup"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Docker image
DOCKER_IMAGE="lmsysorg/sglang:v0.5.6.post1-rocm700-mi35x"

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed or not in PATH${NC}"
    echo "Please install Docker first: https://docs.docker.com/get-docker/"
    exit 1
fi

echo -e "${GREEN}✓${NC} Docker found"

# Check for GPU access
if [ -e /dev/kfd ]; then
    echo -e "${GREEN}✓${NC} ROCm GPU detected (/dev/kfd)"
    GPU_TYPE="rocm"
elif [ -e /dev/nvidia0 ]; then
    echo -e "${GREEN}✓${NC} NVIDIA GPU detected"
    GPU_TYPE="nvidia"
else
    echo -e "${YELLOW}Warning: No GPU device found. Agent will run but cannot benchmark.${NC}"
    GPU_TYPE="none"
fi

# Pull Docker image
echo ""
echo "Pulling Docker image (this may take a while)..."
if docker pull $DOCKER_IMAGE; then
    echo -e "${GREEN}✓${NC} Docker image ready"
else
    echo -e "${RED}Error: Failed to pull Docker image${NC}"
    exit 1
fi

# Make CLI executable
chmod +x mini-kernel

# Verify installation
echo ""
echo "Verifying installation..."

docker run --rm \
    -v "$(pwd)":/workspace \
    -w /workspace \
    $DOCKER_IMAGE \
    python -c "
import sys
sys.path.insert(0, '/workspace')
from mini_kernel import __version__
print(f'Mini-Kernel Agent v{__version__} ready!')
"

echo ""
echo "=============================================="
echo -e "${GREEN}  Setup Complete!${NC}"
echo "=============================================="
echo ""
echo "Quick start:"
echo "  ./mini-kernel optimize examples/add_kernel/kernel.py --gpu 0"
echo ""
echo "For help:"
echo "  ./mini-kernel --help"
echo ""

