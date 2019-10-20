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
len = w * h
a = convert(Array{Float32}, repeat(1:h, 1, w) + repeat(0.001 * (1:w)', h, 1))
d_a = CuArray(a)
d_b = similar(d_a)

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

#####################

allocateArray = CUDA_ARRAY3D_DESCRIPTOR()
allocateArray.Width = Csize_t(w)
allocateArray.Height = Csize_t(h)
allocateArray.Depth = Csize_t(0)
allocateArray.Format = CU_AD_FORMAT_FLOAT
allocateArray.NumChannels = UInt32(1)
allocateArray.Flags = 0


ref_cuarr = Ref{CUarray}(0)
@apicall(:cuArray3DCreate, (Ptr{CUarray}, Ptr{CUDA_ARRAY3D_DESCRIPTOR}), ref_cuarr, Ref(allocateArray))
cuarr = ref_cuarr[]

###################

resrc_desc2 = CUDA_RESOURCE_DESC()
tex_desc2 = CUDA_TEXTURE_DESC()
resrc_view_desc2 = C_NULL#CUDA_RESOURCE_VIEW_DESC()

resrc_desc2.resType = CU_RESOURCE_TYPE_ARRAY
resrc_desc2.res.array.hArray = cuarr
resrc_desc2.flags = 0x0

tex_desc2.addressMode = fill(CUaddress_mode_enum(CU_TR_ADDRESS_MODE_BORDER), 3)
tex_desc2.filterMode = CU_TR_FILTER_MODE_LINEAR
tex_desc2.flags = @CU_TRSF_NORMALIZED_COORDINATES
# tex_desc2.maxAnisotropy = 1
# tex_desc2.mipmapFilterMode = CU_TR_FILTER_MODE_LINEAR
# tex_desc2.mipmapLevelBias = 0
# tex_desc2.minMipmapLevelClamp = 0
# tex_desc2.maxMipmapLevelClamp = 0
tex_desc2.borderColor = [0.12f0, 0.34f0, 0.56f0, 0.78f0]

ref_texobj2 = Ref{CUtexObject}(0)
@apicall(:cuTexObjectCreate, (Ptr{CUtexObject}, Ptr{CUDA_RESOURCE_DESC}, Ptr{CUDA_TEXTURE_DESC}, Ptr{CUDA_RESOURCE_VIEW_DESC}), ref_texobj2, Ref(resrc_desc2), Ref(tex_desc2), C_NULL)
texobj2 = ref_texobj2[]

# @apicall(:cuTexObjectDestroy, (CUtexObject,), texobj2)




########################





@inline function tex2d(texObject::CUtexObject, x, y)::Tuple{Float32,Float32,Float32,Float32}
    Base.llvmcall(("declare [4 x float] @llvm.nvvm.tex.unified.2d.v4f32.f32(i64, float, float)",
        "%4 =  call [4 x float] @llvm.nvvm.tex.unified.2d.v4f32.f32(i64 %0, float %1, float %2)\nret [4 x float] %4"),
        Tuple{Float32,Float32,Float32,Float32},
        Tuple{Int64,Float32,Float32}, convert(Int64, texObject), convert(Float32, x), convert(Float32, y))
end




function
     kernel_texture_warp_native(texObj, dst, h, w)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    u = Float32(i - 1) / Float32(h - 1);
    v = Float32(j - 1) / Float32(w - 1);
    dst[i,j] = tex2d(texObj, u, v)[1];
    return nothing
end
@cuda threads = (w, h) kernel_texture_warp_native(texobj, d_b, Float32(h), Float32(w))

display(d_b)


nothing


#################



# @inline function tex2d(texObject::CUtexObject, x::Cfloat,y::Cfloat)::Cfloat
#     CUDAnative.@wrap llvm.nvvm.__tex2D_float(texObject::i64, x::float, y::float)::float
# end
       
#     # float4 tmp;
#     # asm volatile ("tex.2d.v4.f32.f32 {%0, %1, %2, %3}, [%4, {%5, %6}];" : "=f"(tmp.x), "=f"(tmp.y), "=f"(tmp.z), "=f"(tmp.w) : "l"(texObject), "f"(x), "f"(y));
#     # *retVal = (float)(tmp.x);
# @inline function tex2d(texObject::CUtexObject, x::Cfloat,y::Cfloat)::Cfloat
#     t1,t2,t3,t4 = (0.0f0, 0.0f0, 0.0f0, 0.0f0)
#     CUDAnative.@asmcall("llvm.nvvm.tex.2d.v4.f32.f32 {%0, %1, %2, %3}, [%4, {%5, %6}];", "=f,=f,=f,=f,l,f,f", true,
#     Nothing, Tuple{Cfloat,Cfloat,Cfloat,Cfloat, Culonglong, Cfloat,Cfloat}, t1,t2,t3,t4, texObject, x, y)
# end

# @inline function tex2d(texObject::CUtexObject, x::Cfloat,y::Cfloat)::Cfloat
#     ret = ccall("llvm.nvvm.tex.2d.v4.f32.f32", llvmcall, Tuple{Cfloat,Cfloat,Cfloat,Cfloat}, (Culonglong, Cfloat,Cfloat), texObject, x, y)
#     return ret[1]
# end

# @inline function tex2d(texObject::CUtexObject, x::Cfloat,y::Cfloat)::Cfloat
#     ret = CUDAnative.@wrap "llvm.nvvm.tex.2d.v4.f32.f32"(texObject::i64, x::float, y::float)::float
#     return ret[1]
# end

# @inline function tex2d(texObject::CUtexObject, x::Cfloat,y::Cfloat)::Cfloat
#     ret = ccall("llvm.nvvm.tex.2d.v4.f32.f32", llvmcall, Tuple{Cfloat,Cfloat,Cfloat,Cfloat}, (Culonglong, Cfloat,Cfloat), texObject, x, y)
#     return ret[1]
# end

# @inline function tex2d(texObject::CUtexObject, x::Cfloat,y::Cfloat)::Cfloat
#     CUDAnative.@wrap "llvm.nvvm.tex.unified.2d.v4f32.f32"(texObject::i64, x::float, y::float)::float
# end