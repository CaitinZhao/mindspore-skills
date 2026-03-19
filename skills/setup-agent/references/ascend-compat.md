# Ascend (MindSpore + CANN) Version Compatibility

## Table of Contents
1. [MindSpore ↔ CANN Matrix](#mindspore--cann-matrix)
2. [CANN ↔ Driver/Firmware Matrix](#cann--driverfirmware-matrix)
3. [Python Version Support](#python-version-support)
4. [Install Commands](#install-commands)

## MindSpore ↔ CANN Matrix

| MindSpore | CANN (Recommended) | CANN (Min) | Python |
|-----------|-------------------|------------|--------|
| 2.5.0     | 8.1.RC1           | 8.0.RC3    | 3.8–3.11 |
| 2.4.1     | 8.0.RC3           | 8.0.RC2    | 3.8–3.11 |
| 2.4.0     | 8.0.RC2           | 8.0.RC1    | 3.8–3.11 |
| 2.3.1     | 8.0.RC1           | 7.3.0      | 3.8–3.10 |
| 2.3.0     | 7.3.0             | 7.1.0      | 3.8–3.10 |
| 2.2.14    | 7.1.0             | 7.0.0      | 3.8–3.10 |
| 2.2.0     | 7.0.0             | 6.3.RC3    | 3.7–3.9  |
| 2.1.0     | 6.3.RC2           | 6.3.RC1    | 3.7–3.9  |
| 2.0.0     | 6.3.RC1           | 6.0.1      | 3.7–3.9  |

> These versions are approximate. Always verify against the official
> MindSpore installation guide at https://www.mindspore.cn/install
> CANN has both Community and Commercial editions — version numbers align
> but licensing differs. Use Community for development/research.

## CANN ↔ Driver/Firmware Matrix

| CANN      | NPU Driver (Min)  | Firmware (Min) | Supported Chips |
|-----------|--------------------|----------------|-----------------|
| 8.1.RC1   | 24.1.rc3           | 7.5.0.1.129    | Ascend 910B/C   |
| 8.0.RC3   | 24.1.rc2           | 7.3.0.1.100    | Ascend 910B/C   |
| 8.0.RC2   | 24.1.rc1           | 7.1.0.6.220    | Ascend 910B     |
| 8.0.RC1   | 23.0.6             | 7.1.0.5.220    | Ascend 910B     |
| 7.3.0     | 23.0.5             | 7.1.0.3.220    | Ascend 910A/B   |
| 7.1.0     | 23.0.3             | 7.1.0.1.220    | Ascend 910A/B   |
| 7.0.0     | 23.0.3             | 7.1.0.1.220    | Ascend 910A/B   |
| 6.3.RC2   | 23.0.RC2           | —              | Ascend 910A     |
| 6.3.RC1   | 23.0.RC1           | —              | Ascend 910A     |
| 6.0.1     | 22.0.4             | —              | Ascend 910A     |

## Python Version Support

MindSpore 2.3+ requires Python 3.8–3.11. MindSpore 2.5+ supports 3.8–3.11.
Python 3.12+ is NOT yet supported by MindSpore on Ascend.

## Install Commands

### MindSpore (Ascend)
```bash
# For CANN 8.0+ / MindSpore 2.4+
pip install mindspore==2.5.0
# Or from the Ascend-specific index:
pip install mindspore-ascend==2.5.0
```

### CANN Toolkit
```bash
# Download from Huawei Ascend community:
# https://www.hiascend.com/software/cann/community
# Then install:
chmod +x Ascend-cann-toolkit_<version>_linux-aarch64.run
./Ascend-cann-toolkit_<version>_linux-aarch64.run --install
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### Environment Variables (CANN)
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# Or add to ~/.bashrc:
# export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/lib64:$LD_LIBRARY_PATH
# export ASCEND_HOME=/usr/local/Ascend/ascend-toolkit/latest
```
