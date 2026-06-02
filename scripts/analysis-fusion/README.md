# analysis-fusion exploration scripts

Small DaphneDSL scripts for watching how the compiler lowers an element-wise op
followed by aggregations. Run any of them with `--explain` to dump the IR at a
given stage:

    daphne --explain property_inference --explain kernels 03_fusion_target.daph

Useful stages, roughly in pipeline order:

    parsing               IR right after parsing
    property_inference    after shapes / sparsity / types are inferred
    select_matrix_repr    after dense vs sparse (CSR) is chosen
    kernels               after ops become concrete kernel calls

Add `--codegen` to go through the MLIR code-generation path instead of the
pre-built kernels, and `--explain codegen` to watch that.
