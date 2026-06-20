
cmake_minimum_required(VERSION 3.20)

# Auto-detect LLVM 22 Clang on Debian/WSL if compiler not explicitly provided or was not found
if(NOT CMAKE_CXX_COMPILER OR CMAKE_CXX_COMPILER MATCHES "NOTFOUND")
  find_program(CLANGXX NAMES clang++-22 clang++ g++ HINTS /usr/bin /usr/local/bin)
  if(NOT CLANGXX)
    message(FATAL_ERROR
      "No C++ compiler found. On Debian/WSL run:\n"
      "  sudo apt install clang-22 lld-22 llvm-22-dev libmlir-22-dev mlir-22-tools\n"
      "Then delete your build directory and re-run cmake.")
  endif()
  set(CMAKE_CXX_COMPILER "${CLANGXX}" CACHE FILEPATH "C++ compiler" FORCE)
endif()
if(NOT CMAKE_C_COMPILER OR CMAKE_C_COMPILER MATCHES "NOTFOUND")
  find_program(CLANGC NAMES clang-22 clang gcc HINTS /usr/bin /usr/local/bin)
  if(CLANGC)
    set(CMAKE_C_COMPILER "${CLANGC}" CACHE FILEPATH "C compiler" FORCE)
  endif()
endif()

project(NPUTileCompiler LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default MLIR/LLVM 22 cmake dir on Debian if not passed via -DMLIR_DIR
if(NOT DEFINED MLIR_DIR)
  set(MLIR_DIR "/usr/lib/llvm-22/lib/cmake/mlir" CACHE PATH "Path to MLIRConfig.cmake")
endif()

find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH ${MLIR_CMAKE_DIR})
include(TableGen)
include(AddLLVM)
include(AddMLIR)

include_directories(${MLIR_INCLUDE_DIRS} include)
add_definitions(${LLVM_DEFINITIONS})

set(TD_DIR ${PROJECT_SOURCE_DIR}/include)

set(LLVM_TARGET_DEFINITIONS ${TD_DIR}/MyOps.td)
mlir_tablegen(MyDialect.h.inc   -gen-dialect-decls -I ${TD_DIR})
mlir_tablegen(MyDialect.cpp.inc -gen-dialect-defs  -I ${TD_DIR})
mlir_tablegen(MyOps.h.inc       -gen-op-decls      -I ${TD_DIR})
mlir_tablegen(MyOps.cpp.inc     -gen-op-defs       -I ${TD_DIR})
add_public_tablegen_target(MyIncGen)

set(LLVM_TARGET_DEFINITIONS ${TD_DIR}/NPUOps.td)
mlir_tablegen(NPUDialect.h.inc   -gen-dialect-decls -I ${TD_DIR})
mlir_tablegen(NPUDialect.cpp.inc -gen-dialect-defs  -I ${TD_DIR})
mlir_tablegen(NPUOps.h.inc       -gen-op-decls      -I ${TD_DIR})
mlir_tablegen(NPUOps.cpp.inc     -gen-op-defs       -I ${TD_DIR})
add_public_tablegen_target(NPUIncGen)

add_library(NPUDialect
  lib/MyDialect.cpp
  lib/MyOps.cpp
  lib/NPUDialect.cpp
  lib/NPUOps.cpp
  lib/LowerMyToNPU.cpp
  lib/NPUTiling.cpp
)
add_dependencies(NPUDialect MyIncGen NPUIncGen)

target_include_directories(NPUDialect PUBLIC ${PROJECT_BINARY_DIR})

target_link_libraries(NPUDialect
  MLIRIR MLIRPass MLIRTransforms MLIRSCFDialect MLIRArithDialect MLIRFuncDialect)

add_executable(my-opt tools/my-opt.cpp)
target_link_libraries(my-opt NPUDialect MLIROptLib MLIRParser MLIRSupport)
