#include "NPUOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace {

// Wraps npu.compute_add in a single-iteration scf.for to demonstrate tiling.
// A real tile pass would parameterize lb/ub/step from the surrounding memref shape.
struct TileNPUComputeAdd : public mlir::OpRewritePattern<npu::NPUComputeAddOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(npu::NPUComputeAddOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // Don't re-tile ops already inside a scf.for (prevents infinite loop)
    if (op->getParentOfType<mlir::scf::ForOp>())
      return mlir::failure();

    mlir::Location loc = op.getLoc();

    // Loop bounds: [0, 1) step 1  (one tile for scalar ops)
    mlir::Value lb   = mlir::arith::ConstantIndexOp::create(rewriter, loc, 0);
    mlir::Value ub   = mlir::arith::ConstantIndexOp::create(rewriter, loc, 1);
    mlir::Value step = mlir::arith::ConstantIndexOp::create(rewriter, loc, 1);

    // Carry the result out through an iter_arg initialised to 0
    mlir::Value initAcc = mlir::arith::ConstantIntOp::create(rewriter, loc, 0, 32);

    auto forOp = mlir::scf::ForOp::create(rewriter, loc, lb, ub, step,
                                           mlir::ValueRange{initAcc});

    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(forOp.getBody());

    mlir::Value tileResult = npu::NPUComputeAddOp::create(
        rewriter, loc, rewriter.getI32Type(), op.getA(), op.getB());
    mlir::scf::YieldOp::create(rewriter, loc, mlir::ValueRange{tileResult});

    rewriter.replaceOp(op, forOp.getResult(0));
    return mlir::success();
  }
};

struct NPUTilingPass
    : public mlir::PassWrapper<NPUTilingPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NPUTilingPass)

  llvm::StringRef getArgument() const override { return "npu-tiling"; }
  llvm::StringRef getDescription() const override {
    return "Tile NPU compute ops with scf.for";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::scf::SCFDialect, mlir::arith::ArithDialect>();
  }

  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<TileNPUComputeAdd>(&getContext());
    if (mlir::failed(mlir::applyPatternsGreedily(getOperation(),
                                                  std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

void registerNPUTilingPass() {
  mlir::PassRegistration<NPUTilingPass>();
}
