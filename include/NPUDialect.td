include "mlir/IR/OpBase.td"

def NPUDialect : Dialect {
  let name = "npu";
  let cppNamespace = "::npu";
}
class NPU_Op<string m> : Op<NPUDialect, m>;
