#include "MyOps.h"
#include "NPUOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace {

struct MyAddToNPUComputeAdd : public mlir::OpRewritePattern<my::MyAddOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(my::MyAddOp op, mlir::PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<npu::NPUComputeAddOp>(
        op, rewriter.getI32Type(), op.getA(), op.getB());
    return mlir::success();
  }
};

struct LowerMyToNPUPass
    : public mlir::PassWrapper<LowerMyToNPUPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerMyToNPUPass)

  llvm::StringRef getArgument() const override { return "lower-my-to-npu"; }
  llvm::StringRef getDescription() const override {
    return "Lower my dialect ops to npu dialect ops";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<npu::NPUDialect>();
  }

  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<MyAddToNPUComputeAdd>(&getContext());
    if (mlir::failed(mlir::applyPatternsGreedily(getOperation(),
                                                         std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

void registerLowerMyToNPUPass() {
  mlir::PassRegistration<LowerMyToNPUPass>();
}
