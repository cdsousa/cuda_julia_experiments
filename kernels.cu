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
