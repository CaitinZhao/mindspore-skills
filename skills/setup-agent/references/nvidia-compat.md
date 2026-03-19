# Nvidia (PyTorch + CUDA) Version Compatibility

## Table of Contents
1. [PyTorch ↔ CUDA Matrix](#pytorch--cuda-matrix)
2. [CUDA ↔ Driver Matrix](#cuda--driver-matrix)
3. [Python Version Support](#python-version-support)
4. [Install Commands](#install-commands)

## PyTorch ↔ CUDA Matrix

| PyTorch | CUDA 11.8 | CUDA 12.1 | CUDA 12.4 | CUDA 12.6 | CUDA 12.8 | CUDA 12.9 | CUDA 13.0 |
|---------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| 2.10.0  |           |           |           | Y         | Y         |           | Y         |
| 2.9.x   |           |           |           | Y         | Y         |           | Y         |
| 2.8.0   |           |           |           | Y         | Y         | Y         |           |
| 2.7.x   | Y         |           |           | Y         | Y         |           |           |
| 2.6.0   | Y         |           | Y         | Y         |           |           |           |
| 2.5.x   | Y         | Y         | Y         |           |           |           |           |
| 2.4.x   | Y         | Y         | Y         |           |           |           |           |
| 2.3.x   | Y         | Y         |           |           |           |           |           |
| 2.2.x   | Y         | Y         |           |           |           |           |           |
| 2.1.x   | Y         | Y         |           |           |           |           |           |
| 2.0.x   | Y         | Y         |           |           |           |           |           |

> Note: CUDA 11.8 support was dropped from PyTorch 2.8+.
> CUDA 12.1 was dropped from PyTorch 2.6+.
> Always verify at https://pytorch.org/get-started/previous-versions/

## Recommended Stacks

| Use Case | PyTorch | CUDA | Min Driver (Linux) |
|----------|---------|------|--------------------|
| Latest stable | 2.10.0 | 12.8 | 570.86.15+ |
| Broad GPU compat | 2.7.x | 11.8 | 520.61.05+ |
| Balanced | 2.9.x | 12.6 | 560.28.03+ |
| Bleeding edge | 2.10.0 | 13.0 | 575.51.03+ |

> PyTorch pip wheels bundle their own CUDA runtime, so you don't strictly
> need the CUDA toolkit installed system-wide — but the Nvidia driver must
> meet the minimum version.

## CUDA ↔ Driver Matrix

| CUDA Toolkit | Min Driver (Linux) | Min Driver (Windows) |
|-------------|-------------------|---------------------|
| 13.0        | 575.51.03+        | 576.02+             |
| 12.9        | 570.86.15+        | 571.14+             |
| 12.8        | 570.86.15+        | 571.14+             |
| 12.6        | 560.28.03+        | 560.70+             |
| 12.5        | 555.42.02+        | 555.85+             |
| 12.4        | 550.54.14+        | 551.61+             |
| 12.3        | 545.23.06+        | 545.84+             |
| 12.2        | 535.54.03+        | 536.25+             |
| 12.1        | 530.30.02+        | 531.14+             |
| 12.0        | 525.60.13+        | 527.41+             |
| 11.8        | 520.61.05+        | 522.06+             |

> Check your driver: `nvidia-smi` (top-right shows driver version).
> Check CUDA runtime: `nvcc --version`.

## Python Version Support

PyTorch 2.5+ supports Python 3.9–3.12.
PyTorch 2.8+ supports Python 3.9–3.13.

## Install Commands

### PyTorch (CUDA)
```bash
# Latest stable (CUDA 12.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# CUDA 12.6
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# CUDA 11.8 (older GPUs)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### CUDA Toolkit
```bash
# Ubuntu/Debian (example for CUDA 12.8)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install cuda-toolkit-12-8

# Verify
nvcc --version
nvidia-smi
```

### Environment Variables (CUDA)
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```
