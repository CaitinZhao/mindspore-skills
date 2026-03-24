# Ascend Runtime Compatibility and Setup Guidance

This reference is the lookup source for `setup-agent`. Use it after collecting
real system evidence. Do not guess compatibility.

## Quick Use

Use this file in this order:

1. Check `Driver / Firmware / CANN Matrix`
2. If the system layer is healthy, choose the framework table:
   - `MindSpore on Ascend`
   - `PyTorch + torch_npu on Ascend`
3. If a required system component is missing, go directly to
   `Official Installation Guides` and `Repair Policy`
4. Resolve framework compatibility from local data first. For PTA only, use the
   upstream `Ascend/pytorch` README as a remote fallback when the local table
   cannot classify the tuple.

Decision rule:
- no NPU card detected: stop immediately and skip later driver/CANN checks
- missing driver or missing CANN: stop at the system layer
- `set_env.sh` cannot be sourced: stop at the system layer
- system layer healthy: continue to `uv` and Python package checks
- unknown version tuple: mark `WARN`, do not silently treat as supported

## Compatibility Source Policy

Use these source rules when deciding whether an installed framework is
compatible with the detected CANN version and whether a replacement can be
recommended.

### MindSpore source policy

- local table only
- do not fetch remote compatibility data for MindSpore during a normal
  `setup-agent` run
- if the local MindSpore table cannot classify the tuple, mark it `WARN` and
  do not auto-remediate

### PTA source policy

Lookup order:

1. `Local PTA Compatibility Table`
2. upstream `Ascend/pytorch` README remote fallback
3. if still unresolved, mark the tuple `WARN` and stop PTA auto-remediation

Remote fallback rules:
- use it only when the detected PTA tuple or target CANN row is not available
  locally
- use it to derive a compatible replacement tuple before recommending package
  replacement
- do not mutate this local reference file during a normal `setup-agent` run
- if the upstream data still leaves Python compatibility unresolved, classify
  that PTA path as `WARN` and tell the user to verify the current PTA release
  notes before installation

## Framework Package Remediation Policy

Use these rules after the system layer is healthy and the target `uv`
environment has been confirmed.

- if a framework package is missing, install the compatible version resolved
  from the current CANN version and Python version inside the selected `uv`
  environment
- if an installed framework package is incompatible, recommend the compatible
  replacement version and ask for confirmation before replacing it
- if a framework import or smoke test fails because a Python dependency is
  clearly missing, install that dependency inside the selected `uv`
  environment and retry the failed check
- if the dependency name cannot be identified with high confidence, stop and
  report the unresolved package name instead of guessing
- never modify the system interpreter
- never install, upgrade, or downgrade driver, firmware, or CANN from
  `setup-agent`

## Driver / Firmware / CANN Matrix

Use this table for bottom-up validation. Driver and firmware are the baseline;
CANN must match them before any framework path is considered.

| CANN | Min Driver | Min Firmware | Supported Chips | Typical Use |
|------|------------|--------------|-----------------|-------------|
| 8.1.RC1 | 24.1.rc3 | 7.5.0.1.129 | Ascend 910B/C | Newer 910B/910C deployments |
| 8.0.RC3 | 24.1.rc2 | 7.3.0.1.100 | Ascend 910B/C | Common production baseline |
| 8.0.RC2 | 24.1.rc1 | 7.1.0.6.220 | Ascend 910B | Transitional release |
| 8.0.RC1 | 23.0.6 | 7.1.0.5.220 | Ascend 910B | Older 910B stacks |
| 7.3.0 | 23.0.5 | 7.1.0.3.220 | Ascend 910A/B | Legacy supported path |
| 7.1.0 | 23.0.3 | 7.1.0.1.220 | Ascend 910A/B | Legacy supported path |

Interpretation:
- `PASS`: driver and firmware meet or exceed the minimum for the detected CANN
- `FAIL`: driver or firmware is below the required minimum
- `WARN`: one of the detected versions is not in the table

Stop conditions:
- no Ascend NPU card is present on the machine
- `npu-smi info` fails and no usable backup evidence exists
- driver version file is missing
- CANN toolkit version file is missing
- `set_env.sh` is missing or cannot be sourced

## MindSpore on Ascend

Use this section only after the system layer is healthy.

| MindSpore | Recommended CANN | Minimum CANN | Python | Recommended Replacement | Typical Use |
|-----------|------------------|--------------|--------|-------------------------|-------------|
| 2.5.0 | 8.1.RC1 | 8.0.RC3 | 3.8-3.11 | 2.5.0 on CANN 8.1.RC1, or 2.4.1 on CANN 8.0.RC3 | Current recommended stable line |
| 2.4.1 | 8.0.RC3 | 8.0.RC2 | 3.8-3.11 | 2.4.1 on CANN 8.0.RC3, or 2.4.0 on CANN 8.0.RC2 | Common production baseline |
| 2.4.0 | 8.0.RC2 | 8.0.RC1 | 3.8-3.11 | 2.4.0 on CANN 8.0.RC2, or 2.3.1 on CANN 8.0.RC1 | Transitional release |
| 2.3.1 | 8.0.RC1 | 7.3.0 | 3.8-3.10 | 2.3.1 on CANN 8.0.RC1, or 2.3.0 on CANN 7.3.0 | Legacy support |
| 2.3.0 | 7.3.0 | 7.1.0 | 3.8-3.10 | 2.3.0 on CANN 7.3.0 | Legacy support |

Validation checklist:
- import succeeds inside the selected `uv` environment
- Python version is within the supported range
- CANN version satisfies at least the minimum
- `ms.set_context(device_target='Ascend')` succeeds after sourcing Ascend env
- if the installed MindSpore version is incompatible but a compatible local
  replacement can be derived, recommend replacement inside the selected `uv`
  environment after user confirmation

Decision rule:
- exact or clearly in-range tuple: `PASS`
- version present but below minimum CANN or Python range: `FAIL`
- version not listed: `WARN`
- installed version incompatible but replacement available locally: `FAIL`
  until the user confirms replacement and the replacement smoke test passes

Official verification:
- https://www.mindspore.cn/install

## PyTorch + torch_npu on Ascend

Use this section only after the system layer is healthy.

### Local PTA Compatibility Table

Use exact CANN-keyed rows first. Normalize package versions before lookup:
- strip a leading `v` if present
- compare `torch` on `major.minor.patch`
- compare `torch_npu` on the full version including `.postN` when present
- treat `torch==2.6.0+cpu` as `2.6.0` for compatibility lookup

| CANN | torch | torch_npu | Python | Github Branch | Typical Use |
|------|-------|-----------|--------|---------------|-------------|
| 8.5.0 | 2.9.0 | 2.9.0 | verify upstream PTA Python table before install | v2.9.0-7.3.0 | Latest published PTA line in the upstream compatibility table |
| 8.5.0 | 2.8.0 | 2.8.0.post2 | 3.9-3.11 | v2.8.0-7.3.0 | Newer Ascend stacks |
| 8.5.0 | 2.7.1 | 2.7.1.post2 | 3.9-3.11 | v2.7.1-7.3.0 | Newer Ascend stacks |
| 8.5.0 | 2.6.0 | 2.6.0.post5 | 3.9-3.11 | v2.6.0-7.3.0 | Newer Ascend stacks |
| 8.3.RC1 | 2.8.0 | 2.8.0 | 3.9-3.11 | v2.8.0-7.2.0 | Current RC line |
| 8.3.RC1 | 2.7.1 | 2.7.1 | 3.9-3.11 | v2.7.1-7.2.0 | Current RC line |
| 8.3.RC1 | 2.6.0 | 2.6.0.post3 | 3.9-3.11 | v2.6.0-7.2.0 | Current RC line |
| 8.3.RC1 | 2.1.0 | 2.1.0.post17 | 3.8-3.11 | v2.1.0-7.2.0 | Legacy compatibility |
| 8.2.RC1 | 2.6.0 | 2.6.0 | 3.9-3.11 | v2.6.0-7.1.0 | Transitional line |
| 8.2.RC1 | 2.5.1 | 2.5.1.post1 | 3.9-3.11 | v2.5.1-7.1.0 | Transitional line |
| 8.2.RC1 | 2.1.0 | 2.1.0.post13 | 3.8-3.11 | v2.1.0-7.1.0 | Legacy compatibility |
| 8.1.RC1 | 2.5.1 | 2.5.1 | 3.9-3.11 | v2.5.1-7.0.0 | Common production baseline |
| 8.1.RC1 | 2.4.0 | 2.4.0.post4 | 3.8-3.11 | v2.4.0-7.0.0 | Common production baseline |
| 8.1.RC1 | 2.3.1 | 2.3.1.post6 | 3.8-3.11 | v2.3.1-7.0.0 | Common production baseline |
| 8.1.RC1 | 2.1.0 | 2.1.0.post12 | 3.8-3.11 | v2.1.0-7.0.0 | Older but still common |
| 8.0.0 | 2.4.0 | 2.4.0.post2 | 3.8-3.11 | v2.4.0-6.0.0 | Stable 8.0 line |
| 8.0.0 | 2.3.1 | 2.3.1.post4 | 3.8-3.11 | v2.3.1-6.0.0 | Stable 8.0 line |
| 8.0.0 | 2.1.0 | 2.1.0.post10 | 3.8-3.11 | v2.1.0-6.0.0 | Stable 8.0 line |
| 8.0.RC3 | 2.4.0 | 2.4.0 | 3.8-3.11 | v2.4.0-6.0.rc3 | RC baseline |
| 8.0.RC3 | 2.3.1 | 2.3.1.post2 | 3.8-3.11 | v2.3.1-6.0.rc3 | RC baseline |
| 8.0.RC3 | 2.1.0 | 2.1.0.post8 | 3.8-3.11 | v2.1.0-6.0.rc3 | RC baseline |
| 8.0.RC2 | 2.3.1 | 2.3.1 | 3.8-3.11 | v2.3.1-6.0.rc2 | Transitional RC line |
| 8.0.RC2 | 2.2.0 | 2.2.0.post2 | 3.8-3.10 | v2.2.0-6.0.rc2 | Transitional RC line |
| 8.0.RC2 | 2.1.0 | 2.1.0.post6 | 3.8-3.11 | v2.1.0-6.0.rc2 | Transitional RC line |
| 8.0.RC1 | 2.2.0 | 2.2.0 | 3.8-3.10 | v2.2.0-6.0.rc1 | Older RC line |
| 8.0.RC1 | 2.1.0 | 2.1.0.post4 | 3.8-3.11 | v2.1.0-6.0.rc1 | Older RC line |
| 7.0.0 | 2.1.0 | 2.1.0 | 3.8-3.11 | v2.1.0-5.0.0 | Legacy support |

Validation checklist:
- `torch` import succeeds
- `torch_npu` import succeeds
- `torch` and `torch_npu` major/minor versions align
- the `torch_npu` build suffix such as `.postN` matches the detected CANN row
- Python version is within the supported range
- a minimal NPU tensor smoke test succeeds after sourcing Ascend env
- if the installed PTA tuple is incompatible but a compatible tuple can be
  derived locally or from the upstream README, recommend replacement inside the
  selected `uv` environment after user confirmation

Decision rule:
- exact local or remote tuple match and compatible Python/CANN tuple: `PASS`
- `torch` and `torch_npu` major/minor mismatch: `FAIL`
- version known but below required CANN/Python range: `FAIL`
- installed tuple incompatible but a compatible target tuple is known: `FAIL`
  until the user confirms replacement and the replacement smoke test passes
- exact tuple not listed locally: try remote fallback before classifying
- exact tuple unresolved after local and remote lookup: `WARN`

The setup-agent should tell the user to verify against the PTA release notes if
the exact tuple is not listed or if the upstream Python support is not yet
published for a new PTA row such as `2.9.0`.

## Detection Hints

Use these hints when classifying evidence gathered by the skill.

### System-layer healthy

Typical signals:
- `npu-smi info` lists one or more NPUs
- `/usr/local/Ascend/driver/version.info` exists
- `/usr/local/Ascend/ascend-toolkit/latest/version.cfg` exists
- `source /usr/local/Ascend/ascend-toolkit/set_env.sh` succeeds

### Installed but unusable

Typical signals:
- version files exist, but `npu-smi info` fails
- version files exist, but sourcing `set_env.sh` fails
- environment variables remain incomplete after sourcing

### Python-layer ready

Typical signals:
- `uv` is directly resolvable from the shell with `command -v uv` and `uv --version`
- selected `uv` environment is known
- framework import succeeds
- smoke test succeeds on Ascend

## Official Installation Guides

Use these links when the setup-agent detects missing or unusable system-layer
components:

- Ascend CANN download portal: https://www.hiascend.com/cann/download
- MindSpore install guide: https://www.mindspore.cn/install
- Ascend CANN community downloads: https://www.hiascend.com/software/cann/community
- Ascend documentation portal: https://www.hiascend.com/document
- `uv` install guide: https://docs.astral.sh/uv/getting-started/installation/

Recommended manual repair order:

1. Install or repair the Ascend driver package that matches the target chip
2. Install the matching CANN toolkit release
3. Source `/usr/local/Ascend/ascend-toolkit/set_env.sh`
4. Re-run `npu-smi info`
5. Return to `uv` and Python package checks only after the system layer is healthy

If the Ascend driver or toolkit is missing, the setup-agent should explicitly
remind the user to start from:
- https://www.hiascend.com/cann/download

## Repair Policy

Allowed automation:
- install `uv` via the official installer
- update shell PATH with user confirmation so `uv --version` works directly
- create or reuse a user-confirmed `uv` environment
- install or replace framework packages inside that environment
- install missing Python packages inside that environment

Forbidden automation:
- auto-install driver
- auto-install firmware
- auto-install CANN toolkit
- install Python packages into the system interpreter

Stop and hand back to the user when:
- no NPU card is detected
- `npu-smi info` fails
- driver or CANN is missing
- `set_env.sh` cannot be sourced
- `uv` cannot be resolved directly from the shell after install and PATH update
- the user has not confirmed which `uv` environment to use
