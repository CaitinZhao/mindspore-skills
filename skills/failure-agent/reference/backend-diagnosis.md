# Backend Diagnosis Checklist

## Platform layer

- Device health checks
- Driver/toolkit compatibility
- Hardware faults (ECC/link/heartbeat)

## Scripts layer

- Wrong device/dtype/shape usage
- Inconsistent env vars or context settings
- Non-deterministic setup differences

## Framework layer

- MindSpore API misuse or unsupported graph/mode pattern
- PTA/torch_npu API misuse and operator registration gaps

## Backend layer

- CANN/ACLNN/CUDA runtime failures
- Distributed communication path failures
- Kernel/operator runtime mismatches
