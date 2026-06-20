# IREE Integration: Buffer Placement and DMA Planning

## 1. Keeping the `linalg` interface for IREE

`linalg` is the correct interface point. IREE handles everything above it:

```
StableHLO / TOSA  (model input)
        ↓  IREE: op fusion, shape analysis
linalg on tensors
        ↓  IREE: tiling (you supply tile sizes via TilingConfig)
tiled linalg + scf.for
        ↓  IREE: bufferization (tensors → memrefs with memory spaces)
linalg on buffers
        ↓  YOUR custom lowering pass
npu.* dialect
        ↓  YOUR binary encoder
npu.bin
```

You plug in at the `linalg on buffers` → `npu.*` step by implementing an IREE **CodegenTarget**
(or HAL executable target). Your lowering maps standard ops like `linalg.matmul`,
`linalg.conv_2d_nhwc_hwcf`, etc. to your NPU instructions — no need to redefine the ops,
just add lowering patterns from them to your dialect.

---

## 2. Buffer address and DMA determination per node

Buffer placement and DMA insertion are solved **statically at compile time** through three
sequential steps. The runtime never decides — it only executes a pre-planned DMA program.

### Step 1 — Assign memory spaces during bufferization

Each `memref` gets a memory space tag based on tensor size and reuse:

```mlir
// Small tensor that fits in L2 and is reused by next op → L2
%a = memref.alloc() : memref<256xf32, #npu.mem<L2>>

// Large tensor that does not fit → DDR
%b = memref.alloc() : memref<8192xf32, #npu.mem<DDR>>
```

Decision rule:

```
tensor_size ≤ L2_capacity AND reused_by_next_node  →  L2
otherwise                                           →  DDR
```

"Reused by next node" is determined by **liveness analysis** — if the producer's output is
consumed before L2 is needed for something else, it stays in L2.

### Step 2 — Insert DMA ops where memory spaces differ

A pass walks the op graph. When a producer writes to space X and a consumer reads from space Y,
a DMA op is inserted:

```mlir
// Producer wrote to DDR, consumer needs L2 → insert load DMA before compute
npu.dma_load %ddr_buf, %l2_buf : memref<256xf32, DDR> → memref<256xf32, L2>
%result = npu.compute_add %l2_buf, %weight : memref<256xf32, L2>

// Result is large → store back to DDR after compute
npu.dma_store %l2_result, %ddr_out : memref<8192xf32, L2> → memref<8192xf32, DDR>
```

The four cases:

| Producer placement | Consumer placement | DMA needed |
| ------------------ | ------------------ | ---------- |
| L2 | L2 | None — pass L2 address directly |
| L2 | DDR | `dma_store` after producer |
| DDR | L2 | `dma_load` before consumer |
| DDR | DDR | `dma_load` + `dma_store` (or direct DDR compute if NPU supports it) |

### Step 3 — Assign physical offsets (buffer allocation)

Each memory space has a pool. A buffer allocation pass assigns byte offsets within each pool
using **liveness-aware packing** — the same algorithm as register allocation:

```
L2 pool  [0x0000 – 0x7FFF]
  %node1_output  @ 0x0000,  size 1024 B,  live: node1 → node2
  %node2_weight  @ 0x0400,  size  512 B,  live: node2 only
  %node2_output  @ 0x0000,  size 1024 B,  live: node2 → node3
                                           ↑ reuses node1's slot (no longer live)

DDR pool  [base address set by driver at runtime]
  %large_tensor  @ offset 0x00000,  size 32 KB
```

At runtime the driver sets:
- `L2_BASE = 0xFF000000` (hardware register)
- `DDR_BASE = DMA buffer allocated by host`

All addresses in the binary are `BASE + compile-time offset`. The compiler emits the offsets;
the runtime supplies the base addresses.

---

## How IREE models this

IREE's `Stream` dialect represents placement and transfers explicitly:

| Stream construct | Role |
| ---------------- | ---- |
| `stream.resource` | A tensor with an affinity (which memory space / device) |
| `stream.async.transfer` | A DMA between two memory spaces |
| `iree_util.buffer` | Handles compile-time offset assignment within each pool |

When implementing a custom IREE HAL target, you receive already-bufferized IR with
`stream.async.transfer` ops already inserted by IREE. You lower those to your
`npu.dma_load` / `npu.dma_store` ops and lower compute ops to `npu.compute_*`.

---

## Note: `scf.for` — Structured Control Flow loop

`scf` stands for **Structured Control Flow** — an MLIR dialect that models loops and conditionals
while preserving high-level structure (as opposed to unstructured branches in the `cf` dialect).

`scf.for` is a counted loop with explicit loop-carried values:

```mlir
%result = scf.for %iv = %lb to %ub step %step
    iter_args(%acc = %init) -> (i32) {
  %val = npu.compute_add %a, %b : i32
  scf.yield %val : i32
}
```

| Part | Meaning |
| ---- | ------- |
| `%iv` | Induction variable (index, incremented each iteration) |
| `%lb`, `%ub`, `%step` | Lower bound, upper bound, step — all `index` type |
| `iter_args(%acc = %init)` | Loop-carried value: `%acc` starts as `%init`, updated by `scf.yield` each iteration |
| `-> (i32)` | Type of the value produced when the loop exits |
| `scf.yield %val` | Passes `%val` as the next iteration's `%acc` (and as the loop result on exit) |

In `NPUTiling.cpp`, the tiling pass wraps `npu.compute_add` in an `scf.for` with bounds `[0, 1)`
step `1` — a single-iteration loop acting as a structural placeholder. In a real tiler the bounds
come from the tensor shape, producing one iteration per tile.

The key reason MLIR uses `scf.for` instead of raw branches: subsequent passes (vectorization,
parallelization, further tiling) can reason about loop structure without reconstructing it from
branch graphs.

---

## Summary of compile-time passes needed

| Pass | Input | Output |
| ---- | ----- | ------ |
| Bufferization | linalg on tensors | linalg on memrefs with memory space tags |
| Liveness analysis | memref liveness graph | live ranges per buffer |
| DMA insertion | memrefs with mismatched memory spaces | explicit `npu.dma_*` ops |
| Buffer allocation | live ranges + pool capacities | byte offsets per buffer |
| Binary encoding | npu.* IR with offsets | npu.bin |
