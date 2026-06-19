
cmake_minimum_required(VERSION 3.20)
project(NPUTileCompiler LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH ${MLIR_CMAKE_DIR})
include(TableGen)
include(AddLLVM)
include(AddMLIR)

include_directories(${MLIR_INCLUDE_DIRS} include)
add_definitions(${LLVM_DEFINITIONS})

set(TD_DIR ${PROJECT_SOURCE_DIR}/include)

mlir_tablegen(MyOps.h.inc -gen-op-decls -I ${TD_DIR} ${TD_DIR}/MyOps.td)
mlir_tablegen(MyOps.cpp.inc -gen-op-defs  -I ${TD_DIR} ${TD_DIR}/MyOps.td)
mlir_tablegen(NPUOps.h.inc -gen-op-decls -I ${TD_DIR} ${TD_DIR}/NPUOps.td)
mlir_tablegen(NPUOps.cpp.inc -gen-op-defs  -I ${TD_DIR} ${TD_DIR}/NPUOps.td)

add_public_tablegen_target(NPUIncGen)

add_library(NPUDialect
  lib/MyDialect.cpp
  lib/MyOps.cpp
  lib/NPUDialect.cpp
  lib/NPUOps.cpp
  lib/LowerMyToNPU.cpp
  lib/NPUTiling.cpp
)
add_dependencies(NPUDialect NPUIncGen)

target_include_directories(NPUDialect PUBLIC ${PROJECT_BINARY_DIR})

target_link_libraries(NPUDialect
  MLIRIR MLIRPass MLIRTransforms MLIRSCFDialect MLIRArithDialect MLIRFuncDialect)

add_executable(my-opt tools/my-opt.cpp)
target_link_libraries(my-opt NPUDialect MLIRParser MLIRSupport)
