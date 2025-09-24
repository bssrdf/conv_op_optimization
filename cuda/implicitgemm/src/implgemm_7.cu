#include <cstdint>
#include <assert.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include "conv2d.h"
/*
    线程通过共享内存交换数据，然后使用高效的条带访问模式协作访问全局内存，加入 bias Epilogue
*/

#define WARP_SIZE 32 

template<int width = WARP_SIZE>
static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, width);
    }
    return x;
}

template <bool norm>
static __global__ void reduce_rows_f32(const float * __restrict__ x, float * __restrict__ dst, const int ncols) {
    const int row = blockIdx.x;
    const int col = threadIdx.x;

    float     sum        = 0.0f;
    const int num_unroll = 8;
    float     temp[num_unroll];
    float     sum_temp[num_unroll] = { 0.0f };
    for (int i = col; i < ncols;) {
        for (int j = 0; j < num_unroll; ++j) {
            if (i < ncols) {
                temp[j] = x[row * ncols + i];
            } else {
                temp[j] = 0;
            }
            i += blockDim.x;
        }
        for (int j = 0; j < num_unroll; ++j) {
            sum_temp[j] += temp[j];
        }
    }
    for (int j = 0; j < num_unroll; ++j) {
        sum += sum_temp[j];
    }

    // sum up partial sums
    sum = warp_reduce_sum(sum);
    if (blockDim.x > WARP_SIZE) {
        assert((blockDim.x <= 1024) && (blockDim.x % WARP_SIZE) == 0);
        __shared__ float s_sum[32];
        const int        warp_id = threadIdx.x / WARP_SIZE;
        const int        lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = sum;
        }
        __syncthreads();
        sum = 0.0f;
        if (lane_id < (static_cast<int>(blockDim.x) / WARP_SIZE)) {
            sum = s_sum[lane_id];
        }
        sum = warp_reduce_sum(sum);
    }

    if (col != 0) {
        return;
    }

    dst[row] = norm ? sum / ncols : sum;
}


__global__ void implgemm(param_t param, const int ks)
{
    __shared__ __align__(16 * 1024) char smem[24 * 1024];
    float *smemweight = reinterpret_cast<float *>(smem);
    float *smeminput = reinterpret_cast<float *>(smem + 16 * 1024);

    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = gridDim.z;

    // Warp tile
    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    const int mma_tid_x = (lane_id / 2) % 8;
    const int mma_tid_y = (lane_id / 16) * 2 + (lane_id % 2);
    // lds addr
    int weight_lds_addr = (warp_id / 2) * 32 + mma_tid_y * 4;
    int input_lds_addr = (warp_id % 2) * 64 + mma_tid_x * 4;

    int x = bx * 128 + input_lds_addr;
    int y = by * 128 + weight_lds_addr;
    int z = blockIdx.z;

    const unsigned int start_k = z * ks;

    float weight_ldg_reg[4];
    float input_ldg_reg[4];
    // 当前线程处理的数据点在oh、ow上的坐标
    const unsigned int PQ = param.Oh * param.Ow;
    const unsigned int n = (bx * 128 + tx / 2 ) / PQ;
    const unsigned int npq_res = (bx * 128 + tx / 2 ) % PQ;
    int posh_ori = (npq_res / param.Ow) * param.u - param.p;
    int posw_ori = (npq_res % param.Ow) * param.v - param.q;

    
    int inOffset = n * param.c * param.h * param.w;
    int weiOffset = (by * 128 + tx / 8 * 4) * param.c * param.r * param.s;
    int inChannelOffset = param.c * param.w;
    // int weightChannelOffset = param.r * param.s;
    int weightKOffset = param.c * param.r * param.s;

    // sts addr
    int weight_sts_addr = (tx % 8) * 132 +
                          (tx / 8) * 4;
    int input_sts_addr = tx / 2 + (tx % 2) * 128 * 4;

    int write_flag = 1;
    float weight_frag[2][8];
    float input_frag[2][8];
    float output_frag[8][8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
#pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            output_frag[i][j] = 0;
        }
    }
// ldg
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        if (tx % 8 < weightKOffset && by * 128 + tx / 8 * 4 + i < param.k)
        {
            weight_ldg_reg[i] = param.weight[weiOffset + start_k + tx % 8 + i * weightKOffset];
            // if(tx == 0 && bx == 0 && by == 0 && z == 0)
            // {
            //     printf("weight_ldg_reg:%d,%f, %d, %d, %d\n",  i, weight_ldg_reg[i], 
            //         weiOffset, weightKOffset,
            //         weiOffset + tx % 8 + i * weightKOffset);
            // }
        }
        else
        {
            weight_ldg_reg[i] = 0.0;
        }
    }

    // int curC = (tx / 32) / (param.r * param.s);             // channel offset
    // int curR = ((tx / 32) % (param.r * param.s)) / param.s; // kernel r offset
    // int curS = ((tx / 32) % (param.r * param.s)) % param.s; // kernel s offset

    int curR = (start_k + (tx % 2) * 4) / (param.s * param.c);             // channel offset
    int curS = ((start_k + (tx % 2) * 4) % (param.s * param.c)) / param.c; // kernel r offset
    int curC = ((start_k + (tx % 2) * 4) % (param.s * param.c)) % param.c; // kernel s offset

    int curH = posh_ori + curR; // input h
    int curW = posw_ori + curS; // input w
    if (curH >= 0 && curW >= 0 && curW < param.w && curH < param.h){
        int inOffsetTmp = curH * inChannelOffset + curW * param.c + curC;
        float4 tmp = reinterpret_cast<float4 *>(&param.input[inOffset + inOffsetTmp])[0];
        input_ldg_reg[0] = tmp.x;
        input_ldg_reg[1] = tmp.y;
        input_ldg_reg[2] = tmp.z;
        input_ldg_reg[3] = tmp.w;
    } else {
#pragma unroll
        for (int i = 0; i < 4; ++i)
            input_ldg_reg[i] = 0.0;
    }

    // sts
    for (int i = 0; i < 4; ++i)
    {
        smemweight[weight_sts_addr + i] = weight_ldg_reg[i];
    }
    for (int i = 0; i < 4; ++i)
    {
        smeminput[input_sts_addr + i * 128] = input_ldg_reg[i];
    }

    __syncthreads();
    // lds
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        weight_frag[0][i] = smemweight[weight_lds_addr + i];
        weight_frag[0][i + 4] = smemweight[weight_lds_addr + i + 16];
    }
    // if(tx == 0 && bx == 0 && by == 0 && z == 0)
    // {
    //     printf("weight_ldg_reg:%f,%f,%f,%f\n",  weight_frag[0][0], weight_frag[0][1], weight_frag[0][2], weight_frag[0][3]);
    //     printf("weight_ldg_reg:%f,%f,%f,%f\n",  weight_frag[0][4], weight_frag[0][5], weight_frag[0][6], weight_frag[0][7]);
    // }
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        input_frag[0][i] = smeminput[input_lds_addr + i];
        input_frag[0][i + 4] = smeminput[input_lds_addr + i + 32];
    }
    // for (int crs = 0; crs < param.r * param.s * param.c; crs += 8)
    for (int crs = start_k; crs < start_k + ks; crs += 8)
    {
        // ldg
        int weiOffsetTmp = crs + 8 + tx % 8;
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            if (weiOffsetTmp < weightKOffset && by * 128 + tx / 8 * 4 + i < param.k)
            {
                weight_ldg_reg[i] = param.weight[weiOffset + weiOffsetTmp + i * weightKOffset];
            }
            else
            {
                weight_ldg_reg[i] = 0.0;
            }
        }
        
        curR = (crs + 8 + tx % 2 * 4) / (param.s * param.c);             // channel offset
        curS = ((crs + 8 + tx % 2 * 4) % (param.s * param.c)) / param.c; // kernel r offset
        curC = ((crs + 8 + tx % 2 * 4) % (param.s * param.c)) % param.c; // kernel s offset

        int curH = posh_ori + curR; // input h
        int curW = posw_ori + curS; // input w
        if (curH >= 0 && curW >= 0 && curW < param.w && curH < param.h){
            int inOffsetTmp = curH * inChannelOffset + curW * param.c + curC;
            float4 tmp = reinterpret_cast<float4 *>(&param.input[inOffset + inOffsetTmp])[0];
            input_ldg_reg[0] = tmp.x;
            input_ldg_reg[1] = tmp.y;
            input_ldg_reg[2] = tmp.z;
            input_ldg_reg[3] = tmp.w;
        } else {
#pragma unroll
            for (int i = 0; i < 4; ++i)
                input_ldg_reg[i] = 0.0;
        }

        int load_flag = write_flag ^ 1;
#pragma unroll
        for (int subcrs = 0; subcrs < 8 - 1; ++subcrs)
        {
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                weight_frag[(subcrs + 1) % 2][i] = smemweight[load_flag * 132 * 8 + weight_lds_addr + (subcrs + 1) * 132 + i];
                weight_frag[(subcrs + 1) % 2][i + 4] = smemweight[load_flag * 132 * 8 + weight_lds_addr + (subcrs + 1) * 132 + i + 16];
            }
            // float* base_ptr = smemweight + load_flag * 132 * 8 + weight_lds_addr + (subcrs + 1) * 132;

            // // first 4 values -> weight_frag[...][0..3]
            // float4 v0 = *reinterpret_cast<const float4*>(base_ptr);

            // // next 4 values (offset +16) -> weight_frag[...][4..7]
            // float4 v1 = *reinterpret_cast<const float4*>(base_ptr + 16);

            // // unpack into weight_frag
            // *reinterpret_cast<float4*>(&weight_frag[(subcrs + 1) % 2][0]) = v0;
            // *reinterpret_cast<float4*>(&weight_frag[(subcrs + 1) % 2][4]) = v1;
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                input_frag[(subcrs + 1) % 2][i] = smeminput[load_flag * 128 * 8 + input_lds_addr + (subcrs + 1) * 128 + i];
                input_frag[(subcrs + 1) % 2][i + 4] = smeminput[load_flag * 128 * 8 + input_lds_addr + (subcrs + 1) * 128 + i + 32];
            }

#pragma unroll
            for (int i = 0; i < 8; ++i)
            {
#pragma unroll
                for (int j = 0; j < 8; ++j)
                {
                    output_frag[i][j] += weight_frag[subcrs % 2][i] * input_frag[subcrs % 2][j];
                }
            }
        }
        // sts
        for (int i = 0; i < 4; ++i)
        {
            smemweight[write_flag * 132 * 8 + weight_sts_addr + i] = weight_ldg_reg[i];
        }
        for (int i = 0; i < 4; ++i)
        {
            smeminput[write_flag * 128 * 8 + input_sts_addr + i * 128] = input_ldg_reg[i];
        }
        __syncthreads();
        write_flag ^= 1;
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            weight_frag[0][i] = smemweight[(load_flag ^ 1) * 132 * 8 + weight_lds_addr + i];
            weight_frag[0][i + 4] = smemweight[(load_flag ^ 1) * 132 * 8 + weight_lds_addr + i + 16];
        }
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            input_frag[0][i] = smeminput[(load_flag ^ 1) * 128 * 8 + input_lds_addr + i];
            input_frag[0][i + 4] = smeminput[(load_flag ^ 1) * 128 * 8 + input_lds_addr + i + 32];
        }
#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
#pragma unroll
            for (int j = 0; j < 8; ++j)
            {
                output_frag[i][j] += weight_frag[1][i] * input_frag[1][j];
            }
        }
    }

    // reuse smem
    float *smemoutput = reinterpret_cast<float *>(smem);
    // float *smembias = reinterpret_cast<float *>(smem + 16 * 1024);

    // bias ldg/sts
    // if (tx < 128)
    // {
    //     smembias[tx] = param.bias[by * 128 + tx];
    // }

    uint32_t output_sts_addr = warp_id * 512 + mma_tid_y * 4 * 8 * 4 + mma_tid_x * 4;
    uint32_t output_lds_addr = warp_id * 512 + lane_id;
    // uint32_t bias_lds_addr = warp_id / 2 * 32;

    uint32_t m_idx = blockIdx.y * 128 + warp_id / 2 * 32;
    uint32_t n_idx = blockIdx.x * 128 + warp_id % 2 * 64 + lane_id;

#pragma unroll
    for (int i = 0; i < 2; ++i)
    {
#pragma unroll
        for (int j = 0; j < 2; ++j)
        {
            __syncthreads();

#pragma unroll
            for (int subi = 0; subi < 4; ++subi)
            {
#pragma unroll
                for (int subj = 0; subj < 4; ++subj)
                {
                    // output sts
                    smemoutput[output_sts_addr + subi * 8 * 4 + subj] = output_frag[i * 4 + subi][j * 4 + subj];
                }
            }
            __syncthreads();

#pragma unroll
            for (int subk = 0; subk < 16; ++subk)
            {
                int outOffset = n * param.k * param.Oh * param.Ow * bz + (m_idx + i * 16 + subk) * bz * param.Oh * param.Ow + (n_idx + j * 32) * bz ;
                if ((m_idx + i * 16 + subk) < param.k && (n_idx + j * 32) < param.Oh * param.Ow)
                    param.interm[outOffset + z] = smemoutput[output_lds_addr + subk * 32];
            }
        }
    }
}
cudaError_t launch_implgemm(param_t param)
{
    unsigned int n = param.n;
    unsigned int c = param.c;
    unsigned int h = param.h;
    unsigned int w = param.w;
    unsigned int k = param.k;
    unsigned int r = param.r;
    unsigned int s = param.s;
    unsigned int u = param.u;
    unsigned int v = param.v;
    unsigned int p = param.p;
    unsigned int q = param.q;

    int outh = (h - r + 2 * p) / u + 1;
    int outw = (w - s + 2 * q) / v + 1;    

    const int ksplit = param.ksplit;
    assert((c*r*s) % ksplit == 0);
    const int splitk = (c*r*s) / ksplit;

    
    const unsigned int nrows = n * k * outh * outw;

    int blockx = ((n * outh * outw + 127) / 128); // blockx  number
    int blocky = (k + 127) / 128;             // blocky  number
    // int blockx = ((outh * outw + 63) / 64); // blockx  number
    // int blocky = (k + 63) / 64;             // blocky  number
    int blockz = ksplit;                           // blockz  number
    // 合并threadx与thready
    int threadx = 256; // threadx number per block
    int thready = 1;   // thready number per block
    int threadz = 1;   // threadz number per block
    dim3 block(threadx, thready, threadz);
    dim3 grid(blockx, blocky, blockz);
    implgemm<<<grid, block>>>(param, splitk);
    const dim3 block_nums(nrows, 1, 1);
    const dim3 block_dims(512, 1, 1);
    reduce_rows_f32<false><<<block_nums, block_dims>>>(param.interm, param.output, ksplit);
    return cudaGetLastError();
}
