# torch_npu Operator Diagnosis Notes

When stack is `pta`, focus on:

- torch/torch_npu version and CANN compatibility
- operator availability and dtype/shape/device constraints
- HCCL/distributed collective behavior
- CANN runtime error context around failing operator

If unresolved, prepare a precise reproduction package for PTA maintainers.
