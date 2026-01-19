#!/bin/bash
# Mini-Kernel Agent - Profiler Installation Script
# Installs AMD ROCm profiler tools for bottleneck analysis

set -e

echo "=============================================="
echo "  Mini-Kernel Agent - Profiler Installation"
echo "=============================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
ROCM_VERSION="${ROCM_VERSION:-6.0}"
INSTALL_DIR="${INSTALL_DIR:-/opt/rocm}"

print_step() {
    echo -e "\n${BLUE}==>${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Detect OS
detect_os() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$ID
        VERSION=$VERSION_ID
    else
        print_error "Cannot detect OS"
        exit 1
    fi
    echo "Detected OS: $OS $VERSION"
}

# Check if running in Docker
check_docker() {
    if [ -f /.dockerenv ]; then
        IN_DOCKER=true
        print_warning "Running inside Docker container"
    else
        IN_DOCKER=false
    fi
}

# Install ROCm profiler tools on Ubuntu/Debian
install_rocm_ubuntu() {
    print_step "Installing ROCm profiler tools for Ubuntu/Debian..."
    
    # Add ROCm repository if not present
    if [ ! -f /etc/apt/sources.list.d/rocm.list ]; then
        print_step "Adding ROCm repository..."
        wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
        echo "deb [arch=amd64] https://repo.radeon.com/rocm/apt/${ROCM_VERSION} ubuntu main" | \
            sudo tee /etc/apt/sources.list.d/rocm.list
        sudo apt-get update
    fi
    
    # Install profiler tools
    print_step "Installing rocprofiler and roctracer..."
    sudo apt-get install -y \
        rocprofiler \
        roctracer \
        rocm-smi-lib \
        rocm-developer-tools \
        2>/dev/null || true
    
    print_success "ROCm profiler tools installed"
}

# Install inside Docker (tools usually pre-installed)
install_docker() {
    print_step "Checking profiler tools in Docker..."
    
    # Check if rocprof is available
    if command -v rocprof &> /dev/null; then
        print_success "rocprof found: $(which rocprof)"
    else
        print_warning "rocprof not found in PATH"
        # Try to find it
        ROCPROF_PATH=$(find /opt -name "rocprof" 2>/dev/null | head -1)
        if [ -n "$ROCPROF_PATH" ]; then
            print_success "Found rocprof at: $ROCPROF_PATH"
            echo "Add to PATH: export PATH=\$PATH:$(dirname $ROCPROF_PATH)"
        fi
    fi
    
    # Check rocm-smi
    if command -v rocm-smi &> /dev/null; then
        print_success "rocm-smi found"
    fi
    
    # Check Python profiling tools
    print_step "Checking Python profiling tools..."
    python3 -c "import torch; print(f'PyTorch {torch.__version__} with profiler support')" 2>/dev/null || \
        print_warning "PyTorch not found"
}

# Install Python profiling dependencies
install_python_tools() {
    print_step "Installing Python profiling tools..."
    
    pip install --quiet \
        py-spy \
        memory-profiler \
        line-profiler \
        2>/dev/null || true
    
    print_success "Python profiling tools installed"
}

# Create profiler wrapper script
create_profiler_wrapper() {
    print_step "Creating profiler wrapper..."
    
    WRAPPER_PATH="$(dirname "$0")/../mini_kernel/profiler_wrapper.sh"
    
    cat > "$WRAPPER_PATH" << 'WRAPPER_EOF'
#!/bin/bash
# Profiler wrapper for Mini-Kernel Agent
# Usage: ./profiler_wrapper.sh <kernel_script.py> [args...]

SCRIPT="$1"
shift

# Detect available profiler
if command -v rocprof &> /dev/null; then
    PROFILER="rocprof"
elif command -v nsys &> /dev/null; then
    PROFILER="nsys"
else
    PROFILER="python"
fi

case $PROFILER in
    rocprof)
        echo "Using ROCm profiler..."
        rocprof --stats --hip-trace --hsa-trace -o profile_output.csv python3 "$SCRIPT" "$@"
        ;;
    nsys)
        echo "Using NVIDIA Nsight Systems..."
        nsys profile -o profile_output python3 "$SCRIPT" "$@"
        ;;
    python)
        echo "Using Python cProfile..."
        python3 -m cProfile -o profile_output.prof "$SCRIPT" "$@"
        ;;
esac

echo "Profile output saved"
WRAPPER_EOF

    chmod +x "$WRAPPER_PATH"
    print_success "Profiler wrapper created at $WRAPPER_PATH"
}

# Verify installation
verify_installation() {
    print_step "Verifying profiler installation..."
    
    echo ""
    echo "Available profiling tools:"
    echo "=========================="
    
    # ROCm tools
    if command -v rocprof &> /dev/null; then
        echo -e "${GREEN}✓${NC} rocprof     - $(rocprof --version 2>&1 | head -1)"
    else
        echo -e "${YELLOW}○${NC} rocprof     - Not found"
    fi
    
    if command -v rocm-smi &> /dev/null; then
        echo -e "${GREEN}✓${NC} rocm-smi    - Available"
    else
        echo -e "${YELLOW}○${NC} rocm-smi    - Not found"
    fi
    
    # NVIDIA tools
    if command -v nsys &> /dev/null; then
        echo -e "${GREEN}✓${NC} nsys        - Available"
    else
        echo -e "${YELLOW}○${NC} nsys        - Not found"
    fi
    
    if command -v ncu &> /dev/null; then
        echo -e "${GREEN}✓${NC} ncu         - Available"
    else
        echo -e "${YELLOW}○${NC} ncu         - Not found"
    fi
    
    # Python tools
    echo ""
    echo "Python profiling:"
    python3 -c "
import sys
tools = []
try:
    import torch
    if hasattr(torch.profiler, 'profile'):
        tools.append('torch.profiler')
except: pass
try:
    import cProfile
    tools.append('cProfile')
except: pass
try:
    import memory_profiler
    tools.append('memory_profiler')
except: pass

for t in tools:
    print(f'  ✓ {t}')
if not tools:
    print('  (none found)')
"
    
    echo ""
}

# Main installation flow
main() {
    detect_os
    check_docker
    
    if [ "$IN_DOCKER" = true ]; then
        install_docker
    else
        case $OS in
            ubuntu|debian)
                install_rocm_ubuntu
                ;;
            centos|rhel|fedora)
                print_warning "CentOS/RHEL: Please install ROCm manually"
                print_warning "See: https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html"
                ;;
            *)
                print_warning "Unknown OS: $OS"
                print_warning "Please install ROCm profiler tools manually"
                ;;
        esac
    fi
    
    install_python_tools
    create_profiler_wrapper
    verify_installation
    
    echo ""
    echo "=============================================="
    echo -e "${GREEN}  Profiler Installation Complete!${NC}"
    echo "=============================================="
    echo ""
    echo "Usage in Mini-Kernel Agent:"
    echo "  The agent automatically uses available profilers."
    echo ""
    echo "Manual profiling:"
    echo "  rocprof --stats python3 your_kernel.py"
    echo "  python3 -m torch.profiler your_kernel.py"
    echo ""
}

# Run main
main "$@"

