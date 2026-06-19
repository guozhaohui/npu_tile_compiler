def NPU_Dialect : Dialect { let name = "npu"; let cppNamespace = "::npu"; }
class NPU_Op<string m> : Op<NPU_Dialect, m>;
