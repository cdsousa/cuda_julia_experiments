import Pkg; Pkg.activate("."); Pkg.instantiate()

using CUDAdrv, CUDAnative, CuArrays

using CUDAdrv: @apicall





const CUarray_format = Cuint
const CU_AD_FORMAT_FLOAT = (Cuint)(32)



w,h = 128,128
texmem = reshape(LinRange(0.0f0,1.0f0,w*h),w,h)
d_texmem = CuArray(texmem)


texref = Ref{CuPtr{Cvoid}}()
@apicall(:cuTexRefCreate, (Ptr{CuPtr{Cvoid}},), texref)

ByteOffset = Ref{Csize_t}()
# CUresult cuTexRefSetAddress ( size_t* ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, size_t bytes )
@apicall(:cuTexRefSetAddress, ( Ptr{Csize_t}, CuPtr{Cvoid}, CuPtr{Cvoid}, Csize_t ), ByteOffset, texref[], d_texmem.buf.ptr, sizeof(d_texmem) )

desc = Ref((;Width=Csize_t(w), Height=Csize_t(h), Format=CUarray_format(CU_AD_FORMAT_FLOAT), NumChannels=Cuint(1)))
# CUresult cuTexRefSetAddress2D ( CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR* desc, CUdeviceptr dptr, size_t Pitch )
@apicall(:cuTexRefSetAddress2D, ( CuPtr{Cvoid}, Ptr{Cvoid}, CuPtr{Cvoid}, Csize_t ), texref[], desc, d_texmem.buf.ptr, w*4)

p = Ref{CuPtr{Cvoid}}()
@apicall(:cuTexRefGetAddress, (Ptr{CuPtr{Cvoid}}, CuPtr{Cvoid}), p, texref[])
texref_ptr = p[]



############



struct CuTexture; texref::CuPtr{Cvoid}; end

d_tex = CuTexture(texref[])



##########



cucode = """
    extern "C" {

    __global__ void kernel_vadd(const float *a, const float *b, float *c)
    {
        int i = blockIdx.x *blockDim.x + threadIdx.x;
        c[i] = a[i] + b[i];
    }

    __global__ void kernel_texture_warp(const float *src, float *dst)
    {
        int i = blockIdx.x *blockDim.x + threadIdx.x;
        dst[i] = src[i];
    }

    }
    """

write("kernels.cu", cucode)

run(`nvcc -ptx kernels.cu`)

###

using Test

using CUDAdrv
include(joinpath(@__DIR__, "cudadrv_array", "array.jl"))   # real applications: use CuArrays.jl

dev = CuDevice(0)
ctx = CuContext(dev)

md = CuModuleFile(joinpath(@__DIR__, "kernels.ptx"))
kernel_texture_warp = CuFunction(md, "kernel_texture_warp")

dims = (32,32)
a = round.(rand(Float32, dims) * 100)
b = similar(a)

d_a = CuTestArray(a)
d_b = CuTestArray(b)

len = prod(dims)
cudacall(kernel_texture_warp, Tuple{CuPtr{Cfloat},CuPtr{Cfloat}}, d_a, d_b; threads=len)

@test a ≈ Array(d_b)

destroy!(ctx)

