import Pkg; Pkg.activate("."); Pkg.instantiate()

using CBindingGen
using CBindingGen.Clang

hdrs = ["/usr/local/cuda-8.0/include/cuda.h"]

defs = [
    "CUdeviceptr",
    "CUarray",
    "CUtexObject",
    "CUresourcetype",
    "CUmipmappedArray",
    "CUDA_RESOURCE_DESC",
    "CUaddress_mode",
    "CUfilter_mode",
    "CUDA_TEXTURE_DESC",
    "CUresourceViewFormat",
    "CUDA_RESOURCE_VIEW_DESC",
    "CU_TRSF_",
    "CUDA_ARRAY_DESCRIPTOR",
]

ctx = ConverterContext() do decl
	name = spelling(decl)
    any(occursin(def, name) for def in defs)
end

parse_headers!(ctx, hdrs, args = ["-std=gnu99", "-DUSE_DEF=1"])
generate(ctx, joinpath(@__DIR__, "gen"), "cudatex")





using CBinding

CBinding.@ctypedef size_t Base.Csize_t
CBinding.@ctypedef CUmipmappedArray_st Base.Cvoid
CBinding.@ctypedef CUarray_st Base.Cvoid
include("gen/cudatex-atcompile_typedefs.jl")
include("gen/cudatex-atcompile.jl")


#
