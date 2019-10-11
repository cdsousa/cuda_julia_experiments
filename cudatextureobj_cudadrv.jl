import Pkg; Pkg.activate("."); Pkg.instantiate()

using CUDAdrv, CUDAnative, CuArrays

using CUDAdrv: @apicall

dev = CuDevice(0)
ctx = CuContext(dev)

###############################################

using CBinding
# CBinding.alignof(::Type{Val{:native}}, ::Type{CuPtr{Cvoid}}) = CBinding.alignof(Val{:native}, Ptr{Cvoid})


CBinding.@ctypedef size_t Base.Csize_t
CBinding.@ctypedef CUmipmappedArray_st Base.Cvoid
CBinding.@ctypedef CUarray_st Base.Cvoid
include("gen/cudatex-atcompile_typedefs.jl")
include("gen/cudatex-atcompile.jl")

#############################



w, h = 32, 32
len = w*h
a = convert(Array{Float32}, repeat(1:h, 1, w) + repeat(0.001*(1:w)', h, 1))
d_a = CuArray(a)
b = similar(a)
d_b = CuArray(b)

######

resrc_desc = CUDA_RESOURCE_DESC()
tex_desc = CUDA_TEXTURE_DESC()
resrc_view_desc = C_NULL#CUDA_RESOURCE_VIEW_DESC()

resrc_desc.resType = CU_RESOURCE_TYPE_PITCH2D
resrc_desc.res.pitch2D.devPtr = d_a.buf.ptr
resrc_desc.res.pitch2D.format = CU_AD_FORMAT_FLOAT
resrc_desc.res.pitch2D.numChannels = 1
resrc_desc.res.pitch2D.width = w
resrc_desc.res.pitch2D.height = h
resrc_desc.res.pitch2D.pitchInBytes = w * 4
resrc_desc.flags = 0x0

tex_desc.addressMode = fill(CUaddress_mode_enum(CU_TR_ADDRESS_MODE_BORDER), 3)
tex_desc.filterMode = CU_TR_FILTER_MODE_LINEAR
tex_desc.flags = @CU_TRSF_NORMALIZED_COORDINATES
# tex_desc.maxAnisotropy = 1
# tex_desc.mipmapFilterMode = CU_TR_FILTER_MODE_LINEAR
# tex_desc.mipmapLevelBias = 0
# tex_desc.minMipmapLevelClamp = 0
# tex_desc.maxMipmapLevelClamp = 0
tex_desc.borderColor = [0.12f0, 0.34f0, 0.56f0, 0.78f0]



# cuTexObjectCreate ( CUtexObject* pTexObject, const CUDA_RESOURCE_DESC* pResDesc, const CUDA_TEXTURE_DESC* pTexDesc, const CUDA_RESOURCE_VIEW_DESC* pResViewDesc )

ref_texobj = Ref{CUtexObject}(0)
@apicall(:cuTexObjectCreate, (Ptr{CUtexObject}, Ptr{CUDA_RESOURCE_DESC}, Ptr{CUDA_TEXTURE_DESC}, Ptr{CUDA_RESOURCE_VIEW_DESC}), ref_texobj, Ref(resrc_desc), Ref(tex_desc), C_NULL)
texobj = ref_texobj[]

# @apicall(:cuTexObjectDestroy, (CUtexObject,), texobj)




########################




cucode = """
    extern "C" {

    __global__ void kernel_vadd(const float *a, const float *b, float *c)
    {
        int i = blockIdx.x *blockDim.x + threadIdx.x;
        c[i] = a[i] + b[i];
    }

    __global__ void kernel_texture_warp(cudaTextureObject_t texObj, float *dst, int len)
    {
        int i = blockIdx.x *blockDim.x + threadIdx.x;
        float u = float(i)/float(len);
        dst[i] = tex2D<float>(texObj, u, 0.0f );
    }

    }
    """


write("kernels.cu", cucode)

run(`nvcc -ptx kernels.cu`)

###



md = CuModuleFile(joinpath(@__DIR__, "kernels.ptx"))
kernel_texture_warp = CuFunction(md, "kernel_texture_warp")


cudacall(kernel_texture_warp, Tuple{CUtexObject,CuPtr{Cfloat},Cint}, texobj, d_b, len; threads = len)


display(d_b)
