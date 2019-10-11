import Pkg; Pkg.activate("."); Pkg.instantiate()

using CUDAdrv, CUDAnative, CuArrays
using Test


####

cucode = """
    extern "C" {

    __global__ void kernel_vadd(const float *a, const float *b, float *c)
    {
        int i = blockIdx.x *blockDim.x + threadIdx.x;
        c[i] = a[i] + b[i];
    }


    __global__ void kernel_float4(const float4 *a, float4 *b)
    {
        b[0] = a[0];
    }

    __global__ void kernel_uint2(const uint2 *a, uint2 *b)
    {
        b[0] = a[0];
    }

    __global__ void kernel_z(const float4 *a, float *b)
    {
        float4 aa = a[0];
        b[0] = aa.x;
        b[1] = aa.y;
        b[2] = aa.z;
        b[3] = aa.w;
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
vadd = CuFunction(md, "kernel_vadd")

dims = (3,4)
a = round.(rand(Float32, dims) * 100)
b = round.(rand(Float32, dims) * 100)
c = similar(a)

d_a = CuTestArray(a)
d_b = CuTestArray(b)
d_c = CuTestArray(c)

len = prod(dims)
cudacall(vadd, Tuple{CuPtr{Cfloat},CuPtr{Cfloat},CuPtr{Cfloat}}, d_a, d_b, d_c; threads=len)

@test a+b ≈ Array(d_c)

destroy!(ctx)

