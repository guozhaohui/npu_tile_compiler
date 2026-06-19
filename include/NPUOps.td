include "NPUDialect.td"
def NPUComputeAddOp : NPU_Op<"compute_add"> {
  let arguments = (ins I32:$a, I32:$b);
  let results   = (outs I32:$c);
  let assemblyFormat = "$a `,` $b attr-dict `:` type($a)";
}
