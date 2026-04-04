#!/bin/bash
# =============================================================================
# Install ColabFold (AlphaFold2-Multimer) in a new conda environment.
# MSA is generated via the public MMseqs2 API (no local database needed).
# Inference runs on CPU (--cpu flag) since no GPU is available.
#
# Usage: bash 2_setup_colabfold.sh
# =============================================================================

set -e

ENV_NAME="colabfold"
CONDA_BASE=$(conda info --base)

echo "============================================================"
echo "Setting up ColabFold environment: ${ENV_NAME}"
echo "============================================================"

# Create environment with Python 3.10
conda create -n "${ENV_NAME}" python=3.10 -y

# Activate
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "[1/3] Installing ColabFold + AlphaFold2..."
pip install "colabfold[alphafold]@git+https://github.com/sokrypton/ColabFold"

echo "[2/3] Installing CPU-compatible JAX (required for CPU inference)..."
# Uninstall any GPU version and force CPU version
pip install --upgrade "jax[cpu]"

echo "[3/3] Verifying installation..."
colabfold_batch --help | head -5 && echo "ColabFold installed successfully"

echo ""
echo "============================================================"
echo "Setup complete. Activate with:"
echo "  conda activate ${ENV_NAME}"
echo "============================================================"
