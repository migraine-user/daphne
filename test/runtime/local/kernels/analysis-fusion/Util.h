#include <catch.hpp>
#include "runtime/local/kernels/analysis-fusion/AnalysisFlags.h"
template <typename DT, AnalysisFlag... Fs> void checkAnalysisResult(const DT *arg, const DT *exp) {
    using anal_t = AnalysisFlags<Fs...>;
    if constexpr (anal_t::template contains<AnalysisFlag::mean>) {
        CHECK(arg->mean.has_value() == exp->mean.has_value());
        if (arg->mean.has_value())
            CHECK(std::abs(arg->mean.value() - exp->mean.value()) < 1e-2);
    }
    if constexpr (anal_t::template contains<AnalysisFlag::min>) {
        CHECK(arg->min.has_value() == exp->min.has_value());
        if (arg->min.has_value())
            CHECK(arg->min.value() == exp->min.value());
    }
    if constexpr (anal_t::template contains<AnalysisFlag::max>) {
        CHECK(arg->max.has_value() == exp->max.has_value());
        if (arg->max.has_value())
            CHECK(arg->max.value() == exp->max.value());
    }

    if constexpr (anal_t::template contains<AnalysisFlag::sparsity>)
        CHECK(std::abs(arg->sparsity - exp->sparsity) < 1e-10);

    if constexpr (anal_t::template contains<AnalysisFlag::symmetry>)
        CHECK(arg->symmetric == exp->symmetric);

    if constexpr (anal_t::template contains<AnalysisFlag::numDistinct>) {
        CHECK(arg->numDistinct.has_value() == exp->numDistinct.has_value());
        CHECK(arg->numDistinct.value() == exp->numDistinct.value());
    }
}