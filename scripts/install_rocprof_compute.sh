#!/bin/bash
#
# ROCm Compute Profiler (rocprof-compute) Installation Script
# For use with the mini-kernel agent
#
# rocprof-compute (formerly Omniperf) provides higher-level kernel analysis
# including occupancy, HBM usage, register usage to pinpoint bottlenecks.
#
# Reference: https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/install/core-install.html
#

set -e

echo "========================================================================"
echo "  ROCm Compute Profiler (rocprof-compute) Installation"
echo "========================================================================"
echo ""

# Check if running in Docker
if [ -f /.dockerenv ]; then
    echo "[INFO] Running inside Docker container"
    IN_DOCKER=true
else
    echo "[INFO] Running on host system"
    IN_DOCKER=false
fi

# Check ROCm version
if [ -d "/opt/rocm" ]; then
    ROCM_VERSION=$(cat /opt/rocm/.info/version 2>/dev/null || echo "unknown")
    echo "[INFO] ROCm version: $ROCM_VERSION"
else
    echo "[ERROR] ROCm not found at /opt/rocm"
    echo "Please install ROCm first: https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html"
    exit 1
fi

# Check if rocprof-compute is already available
if command -v rocprof-compute &> /dev/null; then
    echo "[INFO] rocprof-compute is already installed"
    rocprof-compute --version 2>/dev/null || true
    
    # Check if dependencies are installed
    echo ""
    echo "[INFO] Checking Python dependencies..."
    DEPS_OK=true
    rocprof-compute --version 2>&1 | grep -q "ERROR" && DEPS_OK=false
    
    if [ "$DEPS_OK" = false ]; then
        echo "[INFO] Installing missing Python dependencies..."
        pip install -r /opt/rocm/libexec/rocprofiler-compute/requirements.txt 2>/dev/null || \
        pip install astunparse==1.6.2 colorlover dash-bootstrap-components dash-svg \
                    kaleido==0.2.1 plotext plotille pymongo tabulate textual \
                    textual_plotext textual-fspicker rich 2>/dev/null
    fi
    
    echo ""
    echo "[SUCCESS] rocprof-compute is ready!"
    exit 0
fi

# Install rocprof-compute based on system
echo ""
echo "[INFO] Installing rocprof-compute..."

# For ROCm 6.2+, rocprof-compute is included
if [ -f "/opt/rocm/bin/rocprof-compute" ]; then
    echo "[INFO] rocprof-compute found in ROCm installation"
    
    # Install Python dependencies
    echo "[INFO] Installing Python dependencies..."
    if [ -f "/opt/rocm/libexec/rocprofiler-compute/requirements.txt" ]; then
        pip install -r /opt/rocm/libexec/rocprofiler-compute/requirements.txt
    else
        # Manual dependency installation
        pip install astunparse==1.6.2 colorlover dash-bootstrap-components dash-svg \
                    kaleido==0.2.1 plotext plotille pymongo tabulate textual \
                    textual_plotext textual-fspicker rich plotly Flask narwhals \
                    mdit-py-plugins linkify-it-py
    fi
else
    # For older ROCm versions, install from source
    echo "[INFO] rocprof-compute not found in ROCm, installing from source..."
    
    INSTALL_DIR="${ROCPROF_COMPUTE_INSTALL_DIR:-/opt/rocprofiler-compute}"
    VERSION="3.3.1"
    
    # Download
    echo "[INFO] Downloading rocprofiler-compute v${VERSION}..."
    cd /tmp
    wget -q "https://github.com/ROCm/rocm-systems/releases/download/v${VERSION}/rocprofiler-compute-v${VERSION}.tar.gz" \
         -O rocprofiler-compute.tar.gz || {
        echo "[ERROR] Failed to download rocprofiler-compute"
        exit 1
    }
    
    tar xfz rocprofiler-compute.tar.gz
    cd rocprofiler-compute-v${VERSION}
    
    # Install Python dependencies
    echo "[INFO] Installing Python dependencies..."
    pip install -r requirements.txt
    
    # Build and install
    echo "[INFO] Building rocprofiler-compute..."
    mkdir -p build && cd build
    cmake -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}/${VERSION} \
          -DPYTHON_DEPS=${INSTALL_DIR}/python-libs ..
    make install
    
    # Add to PATH
    echo "[INFO] Adding to PATH..."
    echo "export PATH=${INSTALL_DIR}/${VERSION}/bin:\$PATH" >> ~/.bashrc
    export PATH=${INSTALL_DIR}/${VERSION}/bin:$PATH
    
    # Cleanup
    cd /tmp
    rm -rf rocprofiler-compute-v${VERSION} rocprofiler-compute.tar.gz
fi

# Verify installation
echo ""
echo "========================================================================"
echo "  Verifying Installation"
echo "========================================================================"

if command -v rocprof-compute &> /dev/null; then
    echo ""
    rocprof-compute --version 2>&1 | head -5
    echo ""
    echo "[SUCCESS] rocprof-compute installed successfully!"
    echo ""
    echo "Usage:"
    echo "  Profile:  rocprof-compute profile --path ./output -- python3 your_script.py"
    echo "  Analyze:  rocprof-compute analyze --path ./output"
    echo ""
else
    echo "[ERROR] Installation failed - rocprof-compute not found in PATH"
    exit 1
fi

