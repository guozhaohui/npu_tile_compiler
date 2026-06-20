# Offline Memory Planning Tool Integration with MLIR

## Concept

Buffer placement and DMA operations determined statically at compile time can be pre-computed
by an offline tool and fed into the MLIR pipeline as input to the bufferization pass.
This avoids re-implementing memory planning inside MLIR.

## Full pipeline

```
Model (ONNX / StableHLO)
        ↓
Offline memory planning tool
  - runs liveness analysis
  - assigns L1/L2/DDR placement
  - computes byte offsets
  - determines which node-to-node transitions need DMA
        ↓
Memory plan file (JSON / protobuf)
  {
    "node": "conv2d_0",
    "input":  { "space": "DDR", "offset": 0x00000 },
    "output": { "space": "L2",  "offset": 0x00000 },
    "dma_before": { "src": "DDR:0x00000", "dst": "L2:0x00000", "size": 1024 }
  }
        ↓
MLIR pipeline
  linalg on tensors
        ↓  custom bufferization pass reads the memory plan file
  linalg on buffers  ← memrefs already carry the right memory space + offset
        ↓  DMA insertion pass reads the memory plan file
  linalg on buffers + npu.dma_* ops  ← DMAs already placed correctly
        ↓  npu lowering pass
  npu.* dialect
        ↓  binary encoder
  npu.bin
```

## Integration point: plan-driven bufferization pass

The custom bufferization pass reads the offline plan and annotates each `memref` with the
pre-decided memory space and byte offset, instead of computing placement itself.

```mlir
// Standard bufferization produces:
%buf = memref.alloc() : memref<256xf32>

// Plan-driven bufferization produces:
%buf = memref.view %l2_pool[%c0] [] : memref<8192xi8, L2> to memref<256xf32, L2>
//                           ^^^  offset taken directly from the plan file
```

A second pass reads the DMA entries from the plan and inserts `npu.dma_*` ops at the
correct positions in the IR.

## Established precedent

This is a well-known pattern used by existing frameworks:

| Framework | Mechanism |
| --------- | --------- |
| **IREE** | "*binding the executable layout*" — memory plan is a contract between compiler and runtime |
| **TVM AOT executor** | Offline pass assigns all buffer offsets; generated C code indexes into a pre-allocated workspace |

## Caveat: consistency between tile sizes and the memory plan

The offline tool must produce a plan consistent with the tile sizes MLIR uses.
Two options:

| Option | Who drives tiling | Trade-off |
| ------ | ----------------- | --------- |
| **1. Fixed tile sizes** | MLIR (pass options) | Simpler integration; tile sizes are inputs to the offline tool |
| **2. Tool-driven tile sizes** | Offline tool outputs tile sizes; MLIR tiling pass is parameterized by them | Better optimization; larger integration effort |

Option 1 is the practical starting point. Option 2 gives better memory utilization once the
integration is stable.

---

## Appendix: AOT Executor

**AOT** stands for **Ahead-Of-Time** executor — as opposed to **JIT** (Just-In-Time).
The model is fully compiled to a static binary at build time, with no runtime compiler or
graph interpreter involved.

### JIT executor (the alternative)

```text
Model → TVM compiler → runtime graph engine → interprets ops one by one at runtime
                                             → allocates buffers at runtime
                                             → dispatches kernels at runtime
```

### AOT executor

```text
Model → TVM compiler → generated C code → compile → binary
                         ↑
                  all buffer offsets, op order,
                  DMA calls baked in at compile time
```

The generated C looks roughly like:

```c
// All buffer offsets are compile-time constants — no runtime allocation
static uint8_t workspace[WORKSPACE_SIZE];

void run_model(float *input, float *output) {
    float *conv_out = (float *)(workspace + 0x0000);  // offset from AOT plan
    float *relu_out = (float *)(workspace + 0x1000);

    npu_dma_load(input, l2_buf, 1024);    // DMA planned offline
    npu_conv2d(l2_buf, conv_out, ...);
    npu_relu(conv_out, relu_out, ...);
    memcpy(output, relu_out, ...);
}
```

### Why AOT matters for NPU

| Property | JIT / graph interpreter | AOT |
| --------- | ----------------------- | --- |
| Runtime overhead | High — op dispatch loop | Zero — direct function calls |
| Memory footprint | Needs full runtime library | Minimal — just the generated binary |
| Buffer allocation | Dynamic (malloc at runtime) | Static (offsets baked in) |
| Suitable for embedded / NPU | No | Yes |
| Debuggability | Harder to inspect | Generated C is readable |

AOT is the standard approach for edge and NPU targets because it eliminates the runtime graph
engine and produces a self-contained binary with pre-planned memory — exactly what the offline
memory planning strategy in this document produces.
