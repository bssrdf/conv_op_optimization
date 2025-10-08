#include <cstdint>
#include <cstdlib>
#include <assert.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include "conv2d.h"
/*
    same as implgemm_8 but removing double buffering
*/

typedef unsigned int uint;
const int WARPSIZE = 32; // warpSize is not constexpr

/*
 * @tparam BM The threadblock size for M dimension SMEM caching.
 * @tparam BN The threadblock size for N dimension SMEM caching.
 * @tparam BK The threadblock size for K dimension SMEM caching.
 * @tparam WM M dim of continuous tile computed by each warp
 * @tparam WN N dim of continuous tile computed by each warp
 * @tparam WMITER The number of subwarp tiling steps in M dimension.
 * @tparam WNITER The number of subwarp tiling steps in N dimension.
 * @tparam TM The per-thread tile size for M dimension.
 * @tparam TN The per-thread tile size for N dimension.
 */
template<const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS,
          const int PAD=4>
__global__ void implgemm(param_t param)
{
    // __shared__ __align__(16 * 1024) char smem[24 * 1024];

    __shared__ char smem[4 * (TM*TN*NUM_THREADS <= (BM * BK +  BK * (BN+PAD)) ? (BM * BK +  BK * (BN+PAD)) : (TM*TN*NUM_THREADS))];
    // __shared__ float smeminput[2 * BM * BK];
    // __shared__ float smemweight[2 * BK * (BN+PAD)];
    float *smemweight = reinterpret_cast<float *>(smem);
    float *smeminput = reinterpret_cast<float *>(smem + BK * (BN+PAD) * 4);

    const uint tx = threadIdx.x;
    const uint bx = blockIdx.x;
    const uint by = blockIdx.y;

    // Warp tile
    const uint lane_id = tx % WARPSIZE;
    const uint warp_id = tx / WARPSIZE;
    const int mma_tid_x = warp_id / (BN / WN); //(lane_id / 2) % 8;
    const int mma_tid_y = warp_id % (BN / WN); //(lane_id / 16) * 2 + (lane_id % 2);

    // lds addr
    // int weight_lds_addr = (warp_id / 2) * 32 + mma_tid_y * 4;
    // int input_lds_addr = (warp_id % 2) * 64 + mma_tid_x * 4;

    // size of the warp subtile
    constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
    constexpr uint WSUBM = WM / WMITER; // 64/2=32
    constexpr uint WSUBN = WN / WNITER; // 32/2=16

    // Placement of the thread in the warp subtile
    // const uint threadIdxInWarp = tx % WARPSIZE;         // [0, 31]
    const uint threadColInWarp = lane_id % (WSUBN / TN); // i%(16/4)
    const uint threadRowInWarp = lane_id / (WSUBN / TN); // i/4

    // int x = bx * BM + input_lds_addr;
    // int y = by * BN + weight_lds_addr;
    int z = blockIdx.z;

    // float weight_ldg_reg[4];
    // float input_ldg_reg[4];
    // 当前线程处理的数据点在oh、ow上的坐标
    // int posh_ori = ((bx * 128 + tx / 2 ) / param.Ow) * param.u - param.p;
    // int posw_ori = ((bx * 128 + tx / 2 ) % param.Ow) * param.v - param.q;
    // int posh_ori = fastdiv(bx * BM + tx / 2, param.OW_fastdiv) * param.u - param.p;
    // int posw_ori = fastmodulo(bx * BM + tx / 2, param.OW_fastdiv) * param.v - param.q;

    int inOffset = z * param.c * param.h * param.w;
    // int weiOffset = (by * BN + tx / 8 * 4) * param.c * param.r * param.s;
    int inChannelOffset = param.c * param.w;
    // int weightChannelOffset = param.r * param.s;
    int weightKOffset = param.c * param.r * param.s;

    // sts addr
    // int weight_sts_addr = (tx % 8) * 132 +
    //                       (tx / 8) * 4;
    float weight_frag[WNITER * TN] = {0.f};
    float input_frag[WMITER * TM] = {0.f};
    float output_frag[WMITER * TM * WNITER * TN] = {0.f};
// #pragma unroll
//     for (int i = 0; i < 8; ++i)
//     {
// #pragma unroll
//         for (int j = 0; j < 8; ++j)
//         {
//             output_frag[i][j] = 0;
//         }
//     }

    // calculating the indices that this thread will load into SMEM
    // we'll load 128bit / 32bit = 4 elements per thread at each step
    const uint innerRowA = tx / (BK / 4);
    const uint innerColA = tx % (BK / 4);
    constexpr uint rowStrideA = (NUM_THREADS * 4) / BK;
    // const uint innerRowB = tx / (BN / 4);
    // const uint innerColB = tx % (BN / 4);
    // constexpr uint rowStrideB = NUM_THREADS / (BN / 4);

// ldg
    const uint weight_sts_addr = innerRowA + innerColA * (BN+PAD) * 4;
    const uint input_sts_addr = innerRowA + innerColA * BM * 4;
    const uint input_lds_addr =  mma_tid_x * WM;
    const uint weight_lds_addr = mma_tid_y * WN;

    for (int crs = 0; crs < param.r * param.s * param.c; crs += BK)
    {
        // ldg
#pragma unroll
        for (uint offset = 0; offset + rowStrideA <= BN; offset += rowStrideA) {
            if (by * BN  + innerRowA + offset < param.k &&  innerColA * 4 + crs < param.c * param.r * param.s){
                float4 tmp = reinterpret_cast<float4 *>(&param.weight[(by * BN + innerRowA + offset) * weightKOffset + innerColA * 4 + crs])[0];
                smemweight[weight_sts_addr + offset +          0] = tmp.x;
                smemweight[weight_sts_addr + offset +   (BN+PAD)] = tmp.y;
                smemweight[weight_sts_addr + offset + 2*(BN+PAD)] = tmp.z;
                smemweight[weight_sts_addr + offset + 3*(BN+PAD)] = tmp.w;
            } else {
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    smemweight[weight_sts_addr + offset + i*(BN+PAD)] = 0.f;
            }
        }

        for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
            const int posh_ori = fastdiv(bx * BM + innerRowA + offset, param.OW_fastdiv) * param.u - param.p;
            const int posw_ori = fastmodulo(bx * BM + innerRowA + offset, param.OW_fastdiv) * param.v - param.q;
            const uint curR = fastdiv(innerColA * 4 + crs,  param.SC_fastdiv);             // channel offset
            const uint curS = fastdiv(fastmodulo(innerColA * 4 + crs, param.SC_fastdiv),  param.C_fastdiv); // kernel r offset
            const uint curC = fastmodulo(fastmodulo(innerColA * 4 + crs, param.SC_fastdiv),  param.C_fastdiv); // kernel r offset

            const int curH = posh_ori + curR; // input h
            const int curW = posw_ori + curS; // input w
            if (curH >= 0 && curW >= 0 && curW < param.w && curH < param.h){
                int inOffsetTmp = curH * inChannelOffset + curW * param.c + curC;
                float4 tmp = reinterpret_cast<float4 *>(&param.input[inOffset + inOffsetTmp])[0];
                smeminput[input_sts_addr + offset +     0] = tmp.x;
                smeminput[input_sts_addr + offset +    BM] = tmp.y;
                smeminput[input_sts_addr + offset +  2*BM] = tmp.z;
                smeminput[input_sts_addr + offset +  3*BM] = tmp.w;
            } else {
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    smeminput[input_sts_addr + offset + i*BM] = 0.f;
            }
        }
        __syncthreads();

        for (int subcrs = 0; subcrs < BK; ++subcrs)
        {
            for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx)
                for (uint i = 0; i < TN; ++i)
                    weight_frag[wSubColIdx * TN + i] = smemweight[subcrs * (BN+PAD) + weight_lds_addr + wSubColIdx * WSUBN + threadColInWarp * TN + i];
            // float* base_ptr = smemweight + load_flag * 132 * 8 + weight_lds_addr + (subcrs + 1) * 132;

            // // first 4 values -> weight_frag[...][0..3]
            // float4 v0 = *reinterpret_cast<const float4*>(base_ptr);

            // // next 4 values (offset +16) -> weight_frag[...][4..7]
            // float4 v1 = *reinterpret_cast<const float4*>(base_ptr + 16);

            // // unpack into weight_frag
            // *reinterpret_cast<float4*>(&weight_frag[(subcrs + 1) % 2][0]) = v0;
            // *reinterpret_cast<float4*>(&weight_frag[(subcrs + 1) % 2][4]) = v1;
// #pragma unroll
//             for (int i = 0; i < 4; ++i)
//             {
//                 input_frag[(subcrs + 1) % 2][i] = smeminput[load_flag * BM * 8 + input_lds_addr + (subcrs + 1) * BM + i];
//                 input_frag[(subcrs + 1) % 2][i + 4] = smeminput[load_flag * BM * 8 + input_lds_addr + (subcrs + 1) * BM + i + 32];
//             }
            for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx)
                for (uint i = 0; i < TM; ++i)
                    input_frag[wSubRowIdx * TM + i] = smeminput[subcrs * BM + input_lds_addr + wSubRowIdx * WSUBM + threadRowInWarp * TM + i];

            // execute warptile matmul
            for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
                for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                    // calculate per-thread results
                    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                            output_frag[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                                        (wSubColIdx * TN) + resIdxN] +=
                                input_frag[wSubRowIdx * TM + resIdxM] *
                                weight_frag[wSubColIdx * TN + resIdxN];
                        }
                    }
                }
            }
        }
        
    }

    // if(tx == 59 && bx == 0 && by == 0 && z == 0){
    //     for (int i = 0; i < WMITER * TM * WNITER * TN; ++i){
    //         printf("%f,",  output_frag[i]);
    //         if((i+1) % (WNITER * TN) == 0)
    //             printf("\n");
    //     }
    //     printf("\n");
    // }
    // if(tx == 59 && bx == 0 && by == 0 && z == 0){
    //     int cnt[3] = {0};
    //     float values[3] = {-1.f};
    //     for (int i = 0; i < WMITER * TM * WNITER * TN; ++i){
    //         for(int j = 0; j < 3; j++){
    //             if (output_frag[i] == values[j]){                    
    //                 cnt[j]++;
    //                 break;                    
    //             } else{
    //                 if (cnt[j] == 0){
    //                     values[j] = output_frag[i];
    //                     cnt[j]++;
    //                     break;
    //                 }
    //             }          
    //         }
    //     }
    //     for(int j = 0; j < 3; j++){
    //         if(values[j] != -1.f)
    //             printf("value: %f, cnt: %d \n", values[j], cnt[j]);
    //     }
    // }

    // reuse smem
    float *smemoutput = reinterpret_cast<float *>(smem);
    // float *smembias = reinterpret_cast<float *>(smem + 16 * 1024);

    // bias ldg/sts
    // if (tx < BN)
    // {
    //     smembias[tx] = param.bias[by * BN + tx];
    // }

    const uint output_lds_addr = warp_id * WSUBM * WSUBN + lane_id;
    const uint output_sts_addr = mma_tid_x * BN / WN * TM * TN * WARPSIZE + mma_tid_y * TM * TN * WARPSIZE +
                         threadColInWarp * TN * WSUBM + threadRowInWarp * TM;
    const uint m_idx = by * BN + mma_tid_y * WN;
    const uint n_idx = bx * BM + mma_tid_x * WM;
#pragma unroll
    for (int i = 0; i < WMITER; ++i)
    {
#pragma unroll
        for (int j = 0; j < WNITER; ++j)
        {
            __syncthreads();

#pragma unroll
            for (int subi = 0; subi < TM; ++subi)
            {
#pragma unroll
                for (int subj = 0; subj < TN; ++subj)
                {
                    // output sts
                    smemoutput[output_sts_addr + subj * WSUBM + subi] =
                        output_frag[(i * TM + subi) * (WNITER * TN) + j * TN + subj];
                }
            }
            __syncthreads();

            for (int subk = 0; subk < TM * TN; ++subk){
                const uint row =  m_idx + j * WSUBN + (lane_id + subk * WARPSIZE) / WSUBM;
                const uint col =  n_idx + i * WSUBM + (lane_id + subk * WARPSIZE) % WSUBM;
                if (row < param.k && col < param.Oh * param.Ow){
                    const uint outOffset = z * param.k * param.Oh * param.Ow +
                            row * param.Oh * param.Ow + col;
                    param.output[outOffset] = smemoutput[output_lds_addr + subk * WARPSIZE];
                }
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

    const uint bm = 64;
    const uint bn = 128;
    const uint bk = 8;

    const uint NUM_THREADS = 128;
    
    const uint wn = 64;
    const uint wm = 32;
    const uint wniter = 2;
    const uint tn = 4;
    const uint tm = 4;
    const uint oniter = 2;
    dim3 blockDim(NUM_THREADS);

    constexpr uint NUM_WARPS = NUM_THREADS / WARPSIZE;

    // warptile in threadblocktile
    static_assert((bn % wn == 0) && (bm % wm == 0), "");
    static_assert((bn / wn) * (bm / wm) == NUM_WARPS, "");

    // threads in warpsubtile
    static_assert(( wm * wn) % (WARPSIZE * tm * tn * wniter) ==  0, "");
    
    constexpr uint wmiter = (wm * wn) / (WARPSIZE * tm * tn * wniter);
    // warpsubtile in warptile
    static_assert((wm % wmiter == 0) && (wn % wniter == 0), "");

    static_assert((NUM_THREADS * 4) % bk == 0,
                    "NUM_THREADS*4 must be multiple of K9_BK to avoid quantization "
                    "issues during GMEM->SMEM tiling (loading only parts of the "
                    "final row of Bs during each iteraion)");
    static_assert((NUM_THREADS * 4) % bn == 0,
                    "NUM_THREADS*4 must be multiple of K9_BN to avoid quantization "
                    "issues during GMEM->SMEM tiling (loading only parts of the "
                    "final row of As during each iteration)");
    static_assert( bn % (16 * tn) == 0,
                    "BN must be a multiple of 16*TN to avoid quantization effects");
    static_assert( bm % (16 * tm) == 0,
                    "BM must be a multiple of 16*TM to avoid quantization effects");
    static_assert(( bm * bk) % (4 * NUM_THREADS) == 0,
                    "BM*BK must be a multiple of 4*256 to vectorize loads");
    static_assert((bn * bk) % (4 * NUM_THREADS) == 0,
                    "BN*BK must be a multiple of 4*256 to vectorize loads");

    // static_assert((tm * tn * wniter * wmiter * NUM_THREADS) % ( 2 * bk * (bm+bn)) == 0,
    //                 "total smem size (in # of floats) must be divisible by size of all regsister files holding outputs ");

    // static_assert(((tm * tn * wniter * wmiter * NUM_THREADS) / ( 2 * bk * (bm+bn))) % oniter == 0, "");

    // static_assert(bk * (bm+bn) >= tm * tn * NUM_THREADS, "shared memory size must be larger than register file size");

    int blockx = ((outh * outw + bm-1) / bm); // blockx  number
    int blocky = (k + bn-1) / bn;             // blocky  number
    // int blockx = ((outh * outw + 63) / 64); // blockx  number
    // int blocky = (k + 63) / 64;             // blocky  number
    int blockz = n;                           // blockz  number
    // 合并threadx与thready
    int threadx = NUM_THREADS; // threadx number per block
    int thready = 1;   // thready number per block
    int threadz = 1;   // threadz number per block
    dim3 block(threadx, thready, threadz);
    dim3 grid(blockx, blocky, blockz);
    implgemm<bm, bn, bk, wm, wn, wniter, tm, tn, NUM_THREADS><<<grid, block>>>(param);
    return cudaGetLastError();
}
