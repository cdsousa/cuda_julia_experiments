import Pkg; Pkg.activate("."); Pkg.instantiate()

using CUDAdrv, CUDAnative, CuArrays

dev = CuDevice(0)
ctx = CuContext(dev)


k(a,b) = (@inbounds a[1] = b[1]; nothing)

struct S; a::Int32; b::Int32; end

t = CuArray([(Float32(0),Int32(0))])

@device_code_llvm @cuda k(t, t)
@device_code_sass @cuda k(t, t)
@device_code_ptx @cuda k(t, t)

s = CuArray([S(0,0)])

@device_code_llvm @cuda k(s, s)
@device_code_sass @cuda k(s, s)
@device_code_ptx @cuda k(s, s)
