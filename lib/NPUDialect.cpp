#include "NPUOps.h"

#define GET_DIALECT_DEFS
#include "NPUDialect.cpp.inc"

void npu::NPUDialect::initialize() {
  addOperations<::npu::NPUComputeAddOp>();
}
