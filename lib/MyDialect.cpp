#include "MyOps.h"

#define GET_DIALECT_DEFS
#include "MyDialect.cpp.inc"

void my::MyDialect::initialize() {
  addOperations<::my::MyAddOp>();
}
