# MLIR C++ OpBuilder Introduction

`OpBuilder` is the main API for constructing MLIR IR programmatically in C++. It maintains an
**insertion point** (which block/position to emit into) and provides `create<OpType>(...)` to build ops.

## Core usage

```cpp
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

mlir::MLIRContext context;
mlir::OpBuilder builder(&context);

// Create a module
auto module = mlir::ModuleOp::create(builder.getUnknownLoc());

// Add a function inside the module
builder.setInsertionPointToEnd(module.getBody());

auto funcType = builder.getFunctionType({builder.getI32Type(),
                                         builder.getI32Type()},
                                        {builder.getI32Type()});
auto func = builder.create<mlir::func::FuncOp>(
    builder.getUnknownLoc(), "main", funcType);

// Add a basic block and emit ops into it
mlir::Block *entry = func.addEntryBlock();
builder.setInsertionPointToStart(entry);

mlir::Value a = entry->getArgument(0);
mlir::Value b = entry->getArgument(1);

// Emit your custom op — equivalent to:  %0 = my.add %a, %b : i32
auto addOp = builder.create<my::MyAddOp>(
    builder.getUnknownLoc(), builder.getI32Type(), a, b);

builder.create<mlir::func::ReturnOp>(
    builder.getUnknownLoc(), mlir::ValueRange{addOp.getResult()});
```

This produces the same IR as a hand-written `.mlir` file.

## Key methods

| Method | Purpose |
|---|---|
| `builder.create<Op>(loc, ...)` | Emit an op at the insertion point |
| `builder.setInsertionPointToStart(block)` | Move insertion into a block |
| `builder.setInsertionPointAfter(op)` | Move insertion after an existing op |
| `builder.getI32Type()` / `getF32Type()` | Get built-in types |
| `builder.getUnknownLoc()` | Placeholder location (use real locs in production) |
| `builder.getContext()` | Access the `MLIRContext` |

## Relationship to rewrite patterns

`mlir::PatternRewriter` (used in pass rewrite patterns) is a subclass of `OpBuilder`.
So `rewriter.replaceOpWithNewOp<npu::NPUComputeAddOp>(...)` in `LowerMyToNPU.cpp` is the
same mechanism, just constrained to a pattern-match context.
