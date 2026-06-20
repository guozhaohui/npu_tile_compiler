
# Tile-based NPU MLIR Compiler (Minimal Example)

This project demonstrates a **minimal but real tile-based NPU compiler** using **MLIR (LLVM 22)**.

## Pipeline

```
my.add
  ↓ lower-my-to-npu
npu.compute_add
  ↓ npu-tiling
scf.for + npu.dma + npu.compute
  ↓ npu-codegen
C-like trace
```

## Requirements (Debian / Ubuntu)

```bash
sudo apt install llvm-22 llvm-22-dev  mlir-22-tools mlir-22-dev clang-22 cmake ninja-build
sudo apt install lsb-release
sudo apt install gpg
sudo apt install libffi-dev libedit-dev zlib1g-dev libzstd-dev libcurl4-openssl-dev

```

## Build

```bash
cmake -B build -G Ninja -DMLIR_DIR=/usr/lib/llvm-22/lib/cmake/mlir
cmake --build build
```

## Run

```bash
./build/my-opt test/test.mlir
./build/my-opt test/test.mlir -lower-my-to-npu -npu-tiling
```

## Output

The final output prints a **C-like NPU execution trace**.
