# Error Code Quick Guide

Use this file as a lightweight routing reference.

## PTA (torch_npu) common patterns

- `ERRxxxxx`: PTA framework/operator/distributed/profiler buckets.
- `100xxx` or related CANN runtime codes: backend/runtime path.
- `HCCL` timeout/connectivity errors: distributed communication path.

## MindSpore common patterns

- Python exceptions (`RuntimeError`, `ValueError`, `TypeError`): often script/framework validation path.
- `ACLNN` code families (`161xxx`, `361xxx`, `561xxx`): Ascend operator API path.
- CANN inner codes (`E*` forms): backend runtime/kernel path.

## Routing hint

- Hardware/ECC/heartbeat/link errors -> Platform
- Code/shape/dtype misuse -> Scripts/Framework
- Backend numeric/alphanumeric codes -> Backend/CANN/ACLNN
