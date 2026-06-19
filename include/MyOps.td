include "MyDialect.td"
def MyAddOp : My_Op<"add"> { let arguments=(ins I32:$a, I32:$b); let results=(outs I32:$c); }
