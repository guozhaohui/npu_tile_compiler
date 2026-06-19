def My_Dialect : Dialect { let name = "my"; let cppNamespace = "::my"; }
class My_Op<string m> : Op<My_Dialect, m>;
