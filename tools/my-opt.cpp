
#include "MyOps.h"
#include "NPUOps.h"
#include "Passes.h"
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

int main(int argc, char **argv) {
  registerLowerMyToNPUPass();
  registerNPUTilingPass();

  mlir::DialectRegistry registry;
  registry.insert<my::MyDialect, npu::NPUDialect,
                  mlir::arith::ArithDialect,
                  mlir::func::FuncDialect,
                  mlir::scf::SCFDialect>();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "NPU tile compiler", registry));
}
