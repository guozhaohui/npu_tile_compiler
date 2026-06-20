# From Lowered MLIR to NPU Binary

After `--lower-my-to-npu` and `--npu-tiling`, the IR is expressed in your custom `npu.*` dialect.
To run on real hardware, several more stages are required. There are two main paths.

---

## Full compilation pipeline overview

```
test.mlir  (my dialect)
    ↓  --lower-my-to-npu
npu dialect IR  (npu.compute_add inside scf.for)
    ↓  --npu-tiling  (already done)
Tiled npu dialect IR
    ↓  buffer allocation
    ↓  scheduling
    ↓  instruction selection
    ↓  binary encoding
NPU binary / firmware image
    ↓  NPU runtime / driver
Execution on hardware
```

---

## Path 1 — LLVM backend (if the NPU has one)

If the NPU vendor provides an LLVM backend (a custom LLVM target), the path is:

```
npu dialect IR
    ↓  lower npu.* → llvm dialect  (add lowering passes to map each op to LLVM intrinsics)
llvm dialect IR
    ↓  mlir-translate --mlir-to-llvmir
LLVM IR (.ll)
    ↓  llc --march=<npu-target> -filetype=obj
Object file (.o)
    ↓  ld / vendor linker
NPU executable binary
```

This is the approach used when the NPU ISA is close to a CPU/DSP (e.g., Qualcomm Hexagon,
Cadence Tensilica, CEVA-X). The NPU vendor ships an LLVM fork with the custom target registered.

---

## Path 2 — Custom codegen (most proprietary NPUs)

Most dedicated neural NPUs do not use LLVM. Instead, the compiler emits a vendor-specific binary
directly from the IR. The stages are:

### 1. Buffer / memory allocation pass

Assign physical memory addresses to every tensor and intermediate result. This decides which
values live in on-chip SRAM, which in external DRAM, and plans DMA transfers between them.

```
npu.compute_add %a, %b : i32
    → alloc: %a @ SRAM[0x0000], %b @ SRAM[0x0004], result @ SRAM[0x0008]
```

Implemented as an MLIR pass that annotates ops or lowers to an explicit `memref` + `dma` dialect.

### 2. Scheduling pass

Order ops to maximize data reuse and pipeline execution across NPU compute engines. May reorder
`scf.for` tiles or insert explicit synchronization barriers between DMA and compute.

### 3. Instruction selection pass

Map each high-level dialect op to one or more NPU ISA instructions. The output is typically an
in-memory instruction sequence or an intermediate representation like a DAG.

```
npu.compute_add  →  NPU_ADD  dst=R0, src0=R1, src1=R2
```

This is usually implemented as a TableGen-driven instruction selector (similar to LLVM's
`SelectionDAG`) or a hand-written pattern-matching emitter.

### 4. Binary encoding

Serialize the instruction sequence to the NPU binary format — the bit-level encoding of opcodes,
register fields, immediate values, and DMA descriptors. The output is a flat binary blob or a
structured firmware image the NPU driver can load.

```cpp
// Example: hand-written emitter
void emitAdd(BinaryStream &out, Reg dst, Reg src0, Reg src1) {
  uint32_t insn = (OPCODE_ADD << 24) | (dst << 16) | (src0 << 8) | src1;
  out.write32(insn);
}
```

---

## Path 3 — Use an existing open-source runtime (recommended for new projects)

Rather than building all of the above from scratch, target an existing runtime that already handles
binary generation and hardware dispatch:

| Framework | Input dialect | Targets |
|-----------|--------------|---------|
| **IREE**  | StableHLO, TOSA, linalg | GPU, CPU, custom NPU via HAL plugin |
| **TVM**   | Relay / TIR  | CPU, GPU, custom accelerators via BYOC |
| **XLA**   | HLO          | TPU, GPU, CPU |
| **ONNX Runtime** | ONNX  | CPU, CUDA, custom EP |

### Glossary

| Term | Full name | What it is |
| ---- | --------- | ---------- |
| **IREE** | Intermediate Representation Execution Environment | Google's end-to-end MLIR-based ML compiler and runtime; targets CPU, GPU, and custom accelerators |
| **StableHLO** | Stable High-Level Operations | A versioned, portable MLIR dialect for ML ops; the interchange format between JAX/TensorFlow/PyTorch and compilers |
| **TOSA** | Tensor Operator Set Architecture | An MLIR dialect defining a small, hardware-friendly set of tensor ops; common target for TFLite/edge models |
| **linalg** | Linear Algebra dialect | An MLIR dialect for structured tensor and buffer computations; the main entry point for loop tiling and fusion in MLIR |
| **HAL** | Hardware Abstraction Layer | IREE's interface between the compiler-generated binary and the physical device driver; you implement one per target |
| **TVM** | Tensor Virtual Machine | An ML compiler stack from Apache; takes models in Relay, compiles to TIR, then to native code |
| **Relay** | — | TVM's high-level graph IR for neural network models (analogous to StableHLO in the MLIR world) |
| **TIR** | Tensor Intermediate Representation | TVM's low-level loop IR (analogous to MLIR's `affine`/`scf` dialects); the stage where tiling and scheduling happen |
| **BYOC** | Bring Your Own Codegen | TVM's plugin mechanism for offloading subgraphs to a custom accelerator backend |
| **XLA** | Accelerated Linear Algebra | Google's domain-specific compiler for linear algebra; originally for TPUs, now also targets GPUs and CPU via MLIR |
| **HLO** | High-Level Optimizer (ops) | XLA's IR; StableHLO is its stable, MLIR-based successor |
| **TPU** | Tensor Processing Unit | Google's custom ASIC for ML workloads; the original target of XLA |
| **ONNX Runtime** | Open Neural Network Exchange Runtime | Microsoft's inference engine for ONNX models; extensible via Execution Providers |
| **EP** | Execution Provider | ONNX Runtime's plugin interface for offloading ops to a custom hardware backend (analogous to IREE HAL) |

For a custom NPU, IREE's **Hardware Abstraction Layer (HAL)** is the most MLIR-native path:
- Implement an IREE HAL driver for the NPU
- IREE handles tiling, buffer management, and scheduling
- You provide the instruction emitter and the device driver interface

---

## What this project would need to extend

This project currently stops at the tiled `npu.*` + `scf.for` IR. To reach a binary:

1. **Add a `--lower-npu-to-llvm` pass** — if targeting an LLVM-based NPU backend
   - Map `npu.compute_add` → an LLVM intrinsic or a runtime library call
   - Lower `scf.for` → `llvm.br` / loop structure (already handled by existing MLIR lowerings)

2. **Or add a custom emitter** — walk the final IR and emit binary opcodes directly

3. **Add a runtime loader** — a small host-side program that:
   - Opens the NPU device via the vendor driver API
   - Uploads input data via DMA
   - Loads and triggers the binary
   - Reads back results

---

## Summary

| Stage | Implemented in this project? |
|-------|------------------------------|
| High-level dialect (`my.*`) | Yes |
| Lowering to NPU dialect | Yes (`--lower-my-to-npu`) |
| Tiling | Yes (`--npu-tiling`) |
| Buffer allocation | No |
| Scheduling | No |
| Instruction selection | No |
| Binary encoding | No |
| Runtime / driver integration | No |
