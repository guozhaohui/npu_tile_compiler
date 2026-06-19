#pragma once
#include "MyDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "MyOps.h.inc"
