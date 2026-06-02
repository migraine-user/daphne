/*
 * Copyright 2026 The DAPHNE Consortium
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <ir/daphneir/Daphne.h>
#include <ir/daphneir/Passes.h>

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/Pass/Pass.h>

using namespace mlir;

// The request pass left a chars_needed note on each producer (what its consumers
// would like). Here we decide what the producer will actually compute and record
// it as chars_compute. For now we fuse everything that was asked for, so this is
// a straight copy -- but keeping it a separate step gives us one place to later
// turn fusion down for cases where it wouldn't pay off.

namespace {

struct FuseCharacteristicsPass : public PassWrapper<FuseCharacteristicsPass, OperationPass<func::FuncOp>> {
    void runOnOperation() final;
    StringRef getArgument() const final { return "fuse-characteristics"; }
    StringRef getDescription() const final { return "Decide which requested stats each producer computes."; }
};

void FuseCharacteristicsPass::runOnOperation() {
    getOperation().walk([&](Operation *op) {
        auto needed = op->getAttrOfType<ArrayAttr>("chars_needed");
        if (!needed || needed.empty())
            return;
        op->setAttr("chars_compute", needed);
    });
}

} // namespace

std::unique_ptr<Pass> daphne::createFuseCharacteristicsPass() { return std::make_unique<FuseCharacteristicsPass>(); }
