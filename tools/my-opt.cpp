
#include <mlir/Tools/mlir-opt/MlirOptMain.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
int main(int argc,char**argv){
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, mlir::scf::SCFDialect>();
  return mlir::asMainReturnCode(
    mlir::MlirOptMain(argc,argv,"NPU tile compiler",registry));
}
