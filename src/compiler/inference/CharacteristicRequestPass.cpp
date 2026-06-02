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
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/Pass/Pass.h>

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>

#include <algorithm>
#include <cstdint>
#include <vector>

using namespace mlir;

// Stat ids, same numbering the runtime uses: 0 = min, 1 = max, 2 = mean.
// When an aggregation reads a matrix, we note on the matrix's producer that
// this stat is wanted, so the producing op can compute it in the same pass.

namespace {

// The stat an all-aggregation asks of its input, or -1 if it's not one we fuse.
int wantedStat(Operation *op) {
    if (isa<daphne::AllAggMinOp>(op))
        return 0;
    if (isa<daphne::AllAggMaxOp>(op))
        return 1;
    if (isa<daphne::AllAggMeanOp>(op))
        return 2;
    return -1;
}

struct CharacteristicRequestPass : public PassWrapper<CharacteristicRequestPass, OperationPass<func::FuncOp>> {
    void runOnOperation() final;
    StringRef getArgument() const final { return "request-characteristics"; }
    StringRef getDescription() const final { return "Note on each matrix which stats its consumers need."; }
};

void CharacteristicRequestPass::runOnOperation() {
    Builder builder(&getContext());

    getOperation().walk([&](Operation *agg) {
        int stat = wantedStat(agg);
        if (stat < 0 || agg->getNumOperands() == 0)
            return;

        // The matrix being aggregated, and the op that produced it.
        Value in = agg->getOperand(0);
        Operation *producer = in.getDefiningOp();
        if (!producer) // came straight from a function argument
            return;

        // Which of the producer's outputs is this value.
        unsigned resIdx = 0;
        for (Value r : producer->getResults()) {
            if (r == in)
                break;
            ++resIdx;
        }
        if (resIdx >= producer->getNumResults())
            return;

        // chars_needed is one list of stat ids per output. Read what's there,
        // add this stat if it's missing, write it back. Several aggregations can
        // share a producer, so we merge instead of overwrite.
        std::vector<std::vector<int32_t>> needed(producer->getNumResults());
        if (auto old = producer->getAttrOfType<ArrayAttr>("chars_needed"))
            for (unsigned i = 0; i < old.size() && i < needed.size(); ++i)
                if (auto inner = dyn_cast<ArrayAttr>(old[i]))
                    for (Attribute a : inner)
                        if (auto n = dyn_cast<IntegerAttr>(a))
                            needed[i].push_back(n.getInt());

        auto &slot = needed[resIdx];
        if (std::find(slot.begin(), slot.end(), stat) == slot.end())
            slot.push_back(stat);

        SmallVector<Attribute> outer;
        for (auto &inner : needed) {
            SmallVector<Attribute> ids;
            for (int32_t s : inner)
                ids.push_back(builder.getI32IntegerAttr(s));
            outer.push_back(builder.getArrayAttr(ids));
        }
        producer->setAttr("chars_needed", builder.getArrayAttr(outer));
    });
}

} // namespace

std::unique_ptr<Pass> daphne::createCharacteristicRequestPass() { return std::make_unique<CharacteristicRequestPass>(); }
