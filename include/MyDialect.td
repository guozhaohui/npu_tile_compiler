include "mlir/IR/OpBase.td"

def MyDialect : Dialect {
  let name = "my";
  let cppNamespace = "::my";
}
class My_Op<string m> : Op<MyDialect, m>;
