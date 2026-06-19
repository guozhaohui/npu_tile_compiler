include "NPUDialect.td"
def NPUComputeAddOp : NPU_Op<"compute_add"> { let arguments=(ins MemRef:$a, MemRef:$b, MemRef:$c); }
