import Pkg; Pkg.activate("."); Pkg.instantiate()

using CUDAdrv, CUDAnative, CuArrays
using Test


function kernel_vadd(a, b, c)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    c[i] = a[i] + b[i]
    return nothing
end



# CUDAdrv functionality: generate and upload data
a = round.(rand(Float32, (3, 4)) * 100)
b = round.(rand(Float32, (3, 4)) * 100)
d_a = CuArray(a)
d_b = CuArray(b)
d_c = similar(d_a)  # output array

# run the kernel and fetch results
# syntax: @cuda [kwargs...] kernel(args...)
@cuda threads=12 kernel_vadd(d_a, d_b, d_c)

# CUDAdrv functionality: download data
# this synchronizes the device
c = Array(d_c)

@test a+b ≈ c