#include <cstdint>
#include <cstdlib>
#include <assert.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include "conv2d.h"
/*
    线程通过共享内存交换数据，然后使用高效的条带访问模式协作访问全局内存，加入 bias Epilogue
    64x64x8 BMxBNxBK
*/

typedef unsigned int uint;
constexpr uint WARPSIZE = 32; // warpSize is not constexpr

static __global__ void reduce_f32(const float * __restrict__ x, float * __restrict__ dst, const int ncols, const int nrows) {
    const int row = blockIdx.x;
    const int col = threadIdx.x;

    float     sum        = 0.0f;
    if (row * blockDim.x + col < ncols) {
        for (int i = 0; i < nrows; ++i){
            sum += x[i * ncols + row * blockDim.x + col];
        }
        dst[row * blockDim.x + col] = sum;
    }
}

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
          // layout: 0, NHWC; 1, NCHW
          const int layout, const bool vec_load_a, const bool vec_load_b,
          const int ksplit, const int PAD=4>
__global__ void implgemm(param_t param)
{
    // __shared__ __align__(16 * 1024) char smem[24 * 1024];

    // __shared__ char smem[4*(2 * BM * BK +  2 * BK * (BN+PAD))];
    __shared__ char smem[4 * (TM*TN*NUM_THREADS <= 2*((BM+PAD) * BK +  BK * (BN+PAD)) ? 2*( (BM+PAD) * BK +  BK * (BN+PAD)) : (TM*TN*NUM_THREADS))];
    // __shared__ float smeminput[2 * BM * BK];
    // __shared__ float smemweight[2 * BK * (BN+PAD)];
    float *smemweight = reinterpret_cast<float *>(smem);
    float *smeminput = reinterpret_cast<float *>(smem + 2 * BK * (BN+PAD) * 4);

    const uint tx = threadIdx.x;
    const uint bx = blockIdx.x;
    const uint by = blockIdx.y;

    const uint PQ = param.Oh * param.Ow;

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

    
    // int inOffset = (ksplit > 0):  z * param.c * param.h * param.w ;
    // int weiOffset = (by * BN + tx / 8 * 4) * param.c * param.r * param.s;
    int inChannelOffset = layout == 0 ? param.c * param.w : param.h * param.w;
    // int weightChannelOffset = param.r * param.s;
    int weightKOffset = param.c * param.r * param.s;

    // uint ks, start_k;

    // if constexpr (ksplit > 0){
    //     const uint ks =  (weightKOffset + ksplit - 1) / ksplit;
    //     const uint start_k = z * ks;
    // } else {
    //     const uint ks = weightKOffset;
    //     const uint start_k = 0;
    // }
    const uint ks =  (ksplit > 0) ? (weightKOffset + ksplit - 1) / ksplit : weightKOffset;
    const uint start_k = (ksplit > 0)? z * ks: 0;
    const uint end_k = min(start_k + ks, weightKOffset);

    // sts addr
    // int weight_sts_addr = (tx % 8) * 132 +
    //                       (tx / 8) * 4;
    int write_flag = 1;
    float weight_frag[2][WNITER * TN] = {0.f};
    float input_frag[2][WMITER * TM] = {0.f};
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
    for (uint offset = 0; offset + rowStrideA <= BN; offset += rowStrideA) {
        if(vec_load_b){
            // if (by * BN  + innerRowA + offset < param.k &&  start_k + innerColA * 4 < param.c * param.r * param.s){
                if (by * BN  + innerRowA + offset < param.k &&   start_k + innerColA * 4 < end_k){
                float4 tmp = reinterpret_cast<float4 *>(&param.weight[(by * BN + innerRowA + offset) * weightKOffset + start_k + innerColA * 4])[0];
                smemweight[weight_sts_addr + offset +          0] = tmp.x;
                smemweight[weight_sts_addr + offset +   (BN+PAD)] = tmp.y;
                smemweight[weight_sts_addr + offset + 2*(BN+PAD)] = tmp.z;
                smemweight[weight_sts_addr + offset + 3*(BN+PAD)] = tmp.w;
            } else {
                #pragma unroll
                for (int i = 0; i < 4; ++i){
                    smemweight[weight_sts_addr + offset + i*(BN+PAD)] = 0.f;
                }
            }
        }else{
            #pragma unroll
            for (int i = 0; i < 4; ++i){
                if (by * BN  + innerRowA + offset < param.k &&  start_k + innerColA * 4 + i < end_k){
                    // float4 tmp = reinterpret_cast<float4 *>(&param.weight[(by * BN + innerRowA + offset) * weightKOffset + innerColA * 4])[0];
                    smemweight[weight_sts_addr + offset + i*(BN+PAD)] = param.weight[(by * BN + innerRowA + offset) * weightKOffset + start_k + innerColA * 4 + i];
                } else {
                    smemweight[weight_sts_addr + offset + i*(BN+PAD)] = 0.f;
                }
            }
        }
    }


    // int curC = (tx / 32) / (param.r * param.s);             // channel offset
    // int curR = ((tx / 32) % (param.r * param.s)) / param.s; // kernel r offset
    // int curS = ((tx / 32) % (param.r * param.s)) % param.s; // kernel s offset

    // int curR = (tx % 2) * 4 / (param.s * param.c);             // channel offset
    // int curS = ((tx % 2) * 4 % (param.s * param.c)) / param.c; // kernel r offset
    // int curC = ((tx % 2) * 4 % (param.s * param.c)) % param.c; // kernel s offset
    
    const uint input_sts_addr = innerRowA + innerColA * (BM+PAD) * 4;
    for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
        int n = (ksplit > 0) ? (bx * BM + innerRowA + offset) / PQ : z;
        const unsigned int npq_res = (bx * BM + innerRowA + offset) % PQ;
        const int posh_ori = fastdiv((ksplit > 0) ? npq_res: bx * BM + innerRowA + offset, param.OW_fastdiv) * param.u - param.p;
        const int posw_ori = fastmodulo((ksplit > 0) ? npq_res: bx * BM + innerRowA + offset, param.OW_fastdiv) * param.v - param.q;
        int inOffset = n * param.c * param.h * param.w ;
        if(vec_load_a){
            const uint cur0 = fastdiv(start_k + innerColA * 4,  
                   layout == 0 ? param.SC_fastdiv : param.RS_fastdiv);             // channel offset
            const uint cur1 = fastdiv(fastmodulo(start_k + innerColA * 4, 
                layout == 0 ? param.SC_fastdiv : param.RS_fastdiv),  
                layout == 0 ? param.C_fastdiv  : param.S_fastdiv); // kernel r offset
            const uint cur2 = fastmodulo(fastmodulo(start_k + innerColA * 4, 
                layout == 0 ? param.SC_fastdiv : param.RS_fastdiv),  
                layout == 0 ? param.C_fastdiv  : param.S_fastdiv); // kernel r offset
            const uint curC = layout == 0 ? cur2 : cur0;
            const uint curR = layout == 0 ? cur0 : cur1;
            const uint curS = layout == 0 ? cur1 : cur2;
            const int curH = posh_ori + curR; // input h
            const int curW = posw_ori + curS; // input w
            if (curH >= 0 && curW >= 0 && curW < param.w && curH < param.h && start_k + innerColA * 4 < end_k){
                int inOffsetTmp = layout == 0 ? 
                                curH * inChannelOffset + curW * param.c + curC:
                                curC * inChannelOffset + curH * param.w + curW;
                float4 tmp = reinterpret_cast<float4 *>(&param.input[inOffset + inOffsetTmp])[0];
                smeminput[input_sts_addr + offset +          0] = tmp.x;
                smeminput[input_sts_addr + offset +     BM+PAD] = tmp.y;
                smeminput[input_sts_addr + offset +  2*(BM+PAD)] = tmp.z;
                smeminput[input_sts_addr + offset +  3*(BM+PAD)] = tmp.w;
            } else {
                #pragma unroll
                for (int i = 0; i < 4; ++i)
                    smeminput[input_sts_addr + offset + i*(BM+PAD)] = 0.f;
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 4; ++i){
                const uint cur0 = fastdiv(start_k + innerColA * 4 + i,  
                    layout == 0 ? param.SC_fastdiv : param.RS_fastdiv);             // channel offset
                const uint cur1 = fastdiv(fastmodulo(start_k + innerColA * 4 + i, 
                    layout == 0 ? param.SC_fastdiv : param.RS_fastdiv),  
                    layout == 0 ? param.C_fastdiv  : param.S_fastdiv); // kernel r offset
                const uint cur2 = fastmodulo(fastmodulo(start_k + innerColA * 4 + i, 
                    layout == 0 ? param.SC_fastdiv : param.RS_fastdiv),  
                    layout == 0 ? param.C_fastdiv  : param.S_fastdiv); // kernel r offset
                const uint curC = layout == 0 ? cur2 : cur0;
                const uint curR = layout == 0 ? cur0 : cur1;
                const uint curS = layout == 0 ? cur1 : cur2;
                // const uint curR = fastdiv(start_k + innerColA * 4 + i,  param.SC_fastdiv);             // channel offset
                // const uint curS = fastdiv(fastmodulo(start_k + innerColA * 4 + i, param.SC_fastdiv),  param.C_fastdiv); // kernel r offset
                // const uint curC = fastmodulo(fastmodulo(start_k + innerColA * 4 + i, param.SC_fastdiv),  param.C_fastdiv); // kernel r offset

                const int curH = posh_ori + curR; // input h
                const int curW = posw_ori + curS; // input w
                if (curH >= 0 && curW >= 0 && curW < param.w && curH < param.h && start_k + innerColA * 4 + i < end_k){
                    // int inOffsetTmp = curH * inChannelOffset + curW * param.c + curC;
                    int inOffsetTmp = layout == 0 ? 
                                curH * inChannelOffset + curW * param.c + curC:
                                curC * inChannelOffset + curH * param.w + curW;
                    smeminput[input_sts_addr + offset + i*(BM+PAD)] = param.input[inOffset + inOffsetTmp];
                } else {
                    smeminput[input_sts_addr + offset + i*(BM+PAD)] = 0.f;
                }
            }
        }
    }

    // sts
    // for (int i = 0; i < 4; ++i)
    // {
    //     smemweight[weight_sts_addr + i*132] = weight_ldg_reg[i];
    // }
    // for (int i = 0; i < 4; ++i)
    // {
    //     smeminput[input_sts_addr + i * 128] = input_ldg_reg[i];
    // }

    __syncthreads();

    // if(tx == 0 && bx == 0 && by == 0 && z == 0){
    //     for(int i=0; i < 128; ++i)
    //         printf("%.2f,",  smeminput[i]);
    //     printf("\n");
    //     for(int i=128; i < 256; ++i)
    //         printf("%.2f,",  smeminput[i]);
    //     printf("\n");
    // }

    // if(tx == 0 && bx == 0 && by == 0 && z == 0){
    //     printf("%u, %u, %u, %u \n",  innerRowA, innerColA, rowStrideA, weight_sts_addr);
    //     for(int i=0; i < 16; ++i)
    //         printf("%f,",  smemweight[i]);
    //     printf("\n");
    //     for(int i=0; i < 16; ++i)
    //         printf("%f,",  param.weight[i*param.c*param.r*param.s]);
    //     printf("\n");
    // }

    // lds
    // int input_lds_addr = (warp_id % 2) * 64 + mma_tid_x * 4;
    const uint input_lds_addr =  mma_tid_x * WM;
#pragma unroll
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx)
#pragma unroll
      for (uint i = 0; i < TM; ++i)
        input_frag[0][wSubRowIdx * TM + i] = smeminput[input_lds_addr + wSubRowIdx * WSUBM +
                               threadRowInWarp * TM + i];

    // int weight_lds_addr = (warp_id / 2) * 32 + mma_tid_y * 4;
    const uint weight_lds_addr = mma_tid_y * WN;
#pragma unroll
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx)
#pragma unroll
      for (uint i = 0; i < TN; ++i)
        weight_frag[0][wSubColIdx * TN + i] = smemweight[weight_lds_addr + wSubColIdx * WSUBN +
                             threadColInWarp * TN + i];

// #pragma unroll
//     for (int i = 0; i < 4; ++i)
//     {
//         weight_frag[0][i] = smemweight[weight_lds_addr + i];
//         weight_frag[0][i + 4] = smemweight[weight_lds_addr + i + 16];
//     }
    // if(tx == 0 && bx == 0 && by == 0 && z == 0)
    // {
    //     printf("weight_ldg_reg:%f,%f,%f,%f\n",  weight_frag[0][0], weight_frag[0][1], weight_frag[0][2], weight_frag[0][3]);
    //     printf("weight_ldg_reg:%f,%f,%f,%f\n",  weight_frag[0][4], weight_frag[0][5], weight_frag[0][6], weight_frag[0][7]);
    // }
// #pragma unroll
//     for (int i = 0; i < 4; ++i)
//     {
//         input_frag[0][i] = smeminput[input_lds_addr + i];
//         input_frag[0][i + 4] = smeminput[input_lds_addr + i + 32];
//     }


    for (int crs = start_k; crs < end_k; crs += BK)
    {
        // ldg
//         if (by * BN + tx / 2 < param.k && tx % 2 * 4 < param.c * param.r * param.s){
//             float4 tmp = reinterpret_cast<float4 *>(&param.weight[by * BN + tx / 2 * weightKOffset + tx % 2 * 4 + crs + 8])[0];
//             weight_ldg_reg[0] = tmp.x;
//             weight_ldg_reg[1] = tmp.y;
//             weight_ldg_reg[2] = tmp.z;
//             weight_ldg_reg[3] = tmp.w;
//         } else {
//  #pragma unroll
//             for (int i = 0; i < 4; ++i)
//                 weight_ldg_reg[i] = 0.0;
//         }
        // curR = (crs + 8 + tx % 2 * 4) / (param.s * param.c);             // channel offset
        // curS = ((crs + 8 + tx % 2 * 4) % (param.s * param.c)) / param.c; // kernel r offset
        // curC = ((crs + 8 + tx % 2 * 4) % (param.s * param.c)) % param.c; // kernel s offset
//         curR = fastdiv(crs + 8 + (tx % 2) * 4,  param.SC_fastdiv);             // channel offset
//         curS = fastdiv(fastmodulo(crs + 8 + (tx % 2) * 4, param.SC_fastdiv),  param.C_fastdiv); // kernel r offset
//         curC = fastmodulo(fastmodulo(crs + 8 + (tx % 2) * 4, param.SC_fastdiv),  param.C_fastdiv); // kernel r offset

//         int curH = posh_ori + curR; // input h
//         int curW = posw_ori + curS; // input w
//         if (curH >= 0 && curW >= 0 && curW < param.w && curH < param.h){
//             int inOffsetTmp = curH * inChannelOffset + curW * param.c + curC;

//             // float4 tmp = reinterpret_cast<float4 *>(&param.input[inOffset + inOffsetTmp])[0];
//             // input_ldg_reg[0] = tmp.x;
//             // input_ldg_reg[1] = tmp.y;
//             // input_ldg_reg[2] = tmp.z;
//             // input_ldg_reg[3] = tmp.w;
//             reinterpret_cast<float4 *>(&input_ldg_reg[0])[0] = reinterpret_cast<float4 *>(&param.input[inOffset + inOffsetTmp])[0];        } else {
// #pragma unroll
//             for (int i = 0; i < 4; ++i)
//                 input_ldg_reg[i] = 0.0;
//         }

        int load_flag = write_flag ^ 1;
#pragma unroll
        for (int subcrs = 0; subcrs < BK - 1; ++subcrs)
        {
// #pragma unroll
//             for (int i = 0; i < 4; ++i)
//             {
//                 weight_frag[(subcrs + 1) % 2][i] = smemweight[load_flag * (BN+4) * 8 + weight_lds_addr + (subcrs + 1) * (BN+4) + i];
//                 weight_frag[(subcrs + 1) % 2][i + 4] = smemweight[load_flag * (BN+4) * 8 + weight_lds_addr + (subcrs + 1) * (BN+4) + i + 16];
//             }
#pragma unroll
            for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx)
#pragma unroll
                for (uint i = 0; i < TN; ++i)
                    weight_frag[(subcrs + 1) % 2][wSubColIdx * TN + i] = smemweight[load_flag * (BN+PAD) * BK +
                        (subcrs + 1) * (BN+PAD) + weight_lds_addr + wSubColIdx * WSUBN + threadColInWarp * TN + i];
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
#pragma unroll
            for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx)
#pragma unroll
                for (uint i = 0; i < TM; ++i)
                    input_frag[(subcrs + 1) % 2][wSubRowIdx * TM + i] = smeminput[load_flag * (BM+PAD) * BK +
                        (subcrs + 1) * (BM+PAD) + input_lds_addr + wSubRowIdx * WSUBM + threadRowInWarp * TM + i];

// #pragma unroll
//             for (int i = 0; i < 8; ++i)
//             {
// #pragma unroll
//                 for (int j = 0; j < 8; ++j)
//                 {
//                     output_frag[i][j] += weight_frag[subcrs % 2][i] * input_frag[subcrs % 2][j];
//                 }
//             }
            // execute warptile matmul
#pragma unroll
            for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
#pragma unroll
                for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                    // calculate per-thread results
#pragma unroll
                    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
#pragma unroll
                        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                            output_frag[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                                        (wSubColIdx * TN) + resIdxN] +=
                                input_frag[subcrs % 2][wSubRowIdx * TM + resIdxM] *
                                weight_frag[subcrs % 2][wSubColIdx * TN + resIdxN];
                            // if(tx == 0 && bx == 0 && by == 0 && z == 0){
                            //     printf("subcrs:%d, i:%d, j:%d, %f * %f = %f, acc = %f\n", subcrs, wSubRowIdx * TM + resIdxM, wSubColIdx * TN + resIdxN,
                            //         input_frag[subcrs % 2][wSubRowIdx * TM + resIdxM],
                            //         weight_frag[subcrs % 2][wSubColIdx * TN + resIdxN],
                            //         input_frag[subcrs % 2][wSubRowIdx * TM + resIdxM] *
                            //         weight_frag[subcrs % 2][wSubColIdx * TN + resIdxN],
                            //         output_frag[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                            //             (wSubColIdx * TN) + resIdxN]);
                            // }
                        }
                    }
                }
            }
        }
        // ldg
#pragma unroll
        for (uint offset = 0; offset + rowStrideA <= BN; offset += rowStrideA) {
            if(vec_load_b){
                if (by * BN  + innerRowA + offset < param.k &&  innerColA * 4 + crs + BK < end_k){
                    float4 tmp = reinterpret_cast<float4 *>(&param.weight[(by * BN + innerRowA + offset) * weightKOffset + innerColA * 4 + crs + BK])[0];
                    smemweight[write_flag * (BN+PAD) * BK + weight_sts_addr + offset +          0] = tmp.x;
                    smemweight[write_flag * (BN+PAD) * BK + weight_sts_addr + offset +   (BN+PAD)] = tmp.y;
                    smemweight[write_flag * (BN+PAD) * BK + weight_sts_addr + offset + 2*(BN+PAD)] = tmp.z;
                    smemweight[write_flag * (BN+PAD) * BK + weight_sts_addr + offset + 3*(BN+PAD)] = tmp.w;
                } else {
                    #pragma unroll
                    for (int i = 0; i < 4; ++i)
                        smemweight[write_flag * (BN+PAD) * BK + weight_sts_addr + offset + i*(BN+PAD)] = 0.f;
                }
            }else{
                #pragma unroll
                for (int i = 0; i < 4; ++i){
                    if (by * BN  + innerRowA + offset < param.k &&  innerColA * 4 + crs + BK + i < end_k){
                        // float4 tmp = reinterpret_cast<float4 *>(&param.weight[(by * BN + innerRowA + offset) * weightKOffset + innerColA * 4 + crs + BK + i])[0];
                        smemweight[write_flag * (BN+PAD) * BK + weight_sts_addr + offset + i*(BN+PAD)] = param.weight[(by * BN + innerRowA + offset) * weightKOffset + innerColA * 4 + crs + BK + i];
                    } else {
                        smemweight[write_flag * (BN+PAD) * BK + weight_sts_addr + offset + i*(BN+PAD)] = 0.f;
                    }
                }
            }
        }
#pragma unroll
        for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
            int n = (ksplit > 0) ? (bx * BM + innerRowA + offset) / PQ : z;
            const unsigned int npq_res = (bx * BM + innerRowA + offset) % PQ;
            const int posh_ori = fastdiv((ksplit > 0) ? npq_res: bx * BM + innerRowA + offset, param.OW_fastdiv) * param.u - param.p;
            const int posw_ori = fastmodulo((ksplit > 0) ? npq_res: bx * BM + innerRowA + offset, param.OW_fastdiv) * param.v - param.q;
            int inOffset = n * param.c * param.h * param.w ;
            if(vec_load_a){
                // const uint curR = fastdiv(innerColA * 4 + crs + BK,  param.SC_fastdiv);             // channel offset
                // const uint curS = fastdiv(fastmodulo(innerColA * 4 + crs + BK, param.SC_fastdiv),  param.C_fastdiv); // kernel r offset
                // const uint curC = fastmodulo(fastmodulo(innerColA * 4 + crs + BK, param.SC_fastdiv),  param.C_fastdiv); // kernel r offset
                const uint cur0 = fastdiv(innerColA * 4 + crs + BK,  
                    layout == 0 ? param.SC_fastdiv : param.RS_fastdiv);             // channel offset
                const uint cur1 = fastdiv(fastmodulo(innerColA * 4 + crs + BK, 
                    layout == 0 ? param.SC_fastdiv : param.RS_fastdiv),  
                    layout == 0 ? param.C_fastdiv  : param.S_fastdiv); // kernel r offset
                const uint cur2 = fastmodulo(fastmodulo(innerColA * 4 + crs + BK, 
                    layout == 0 ? param.SC_fastdiv : param.RS_fastdiv),  
                    layout == 0 ? param.C_fastdiv  : param.S_fastdiv); // kernel r offset
                const uint curC = layout == 0 ? cur2 : cur0;
                const uint curR = layout == 0 ? cur0 : cur1;
                const uint curS = layout == 0 ? cur1 : cur2;
                const int curH = posh_ori + curR; // input h
                const int curW = posw_ori + curS; // input w
                if (curH >= 0 && curW >= 0 && curW < param.w && curH < param.h && innerColA * 4 + crs + BK < end_k){
                    // int inOffsetTmp = curH * inChannelOffset + curW * param.c + curC;
                    int inOffsetTmp = layout == 0 ? 
                                curH * inChannelOffset + curW * param.c + curC:
                                curC * inChannelOffset + curH * param.w + curW;
                    float4 tmp = reinterpret_cast<float4 *>(&param.input[inOffset + inOffsetTmp])[0];
                    smeminput[write_flag * (BM+PAD) * BK + input_sts_addr + offset +     0] = tmp.x;
                    smeminput[write_flag * (BM+PAD) * BK + input_sts_addr + offset +    BM+PAD] = tmp.y;
                    smeminput[write_flag * (BM+PAD) * BK + input_sts_addr + offset +  2*(BM+PAD)] = tmp.z;
                    smeminput[write_flag * (BM+PAD) * BK + input_sts_addr + offset +  3*(BM+PAD)] = tmp.w;
                } else {
    #pragma unroll
                    for (int i = 0; i < 4; ++i)
                        smeminput[write_flag * (BM+PAD) * BK + input_sts_addr + offset + i*(BM+PAD)] = 0.f;
                }
            } else {
                #pragma unroll
                for (int i = 0; i < 4; ++i){
                    // const uint curR = fastdiv(innerColA * 4 + crs + BK + i,  param.SC_fastdiv);             // channel offset
                    // const uint curS = fastdiv(fastmodulo(innerColA * 4 + crs + BK + i, param.SC_fastdiv),  param.C_fastdiv); // kernel r offset
                    // const uint curC = fastmodulo(fastmodulo(innerColA * 4 + crs + BK + i, param.SC_fastdiv),  param.C_fastdiv); // kernel r offset
                    const uint cur0 = fastdiv(innerColA * 4 + crs + BK + i,  
                        layout == 0 ? param.SC_fastdiv : param.RS_fastdiv);             // channel offset
                    const uint cur1 = fastdiv(fastmodulo(innerColA * 4 + crs + BK + i, 
                        layout == 0 ? param.SC_fastdiv : param.RS_fastdiv),  
                        layout == 0 ? param.C_fastdiv  : param.S_fastdiv); // kernel r offset
                    const uint cur2 = fastmodulo(fastmodulo(innerColA * 4 + crs + BK + i, 
                        layout == 0 ? param.SC_fastdiv : param.RS_fastdiv),  
                        layout == 0 ? param.C_fastdiv  : param.S_fastdiv); // kernel r offset
                    const uint curC = layout == 0 ? cur2 : cur0;
                    const uint curR = layout == 0 ? cur0 : cur1;
                    const uint curS = layout == 0 ? cur1 : cur2;
                    const int curH = posh_ori + curR; // input h
                    const int curW = posw_ori + curS; // input w
                    if (curH >= 0 && curW >= 0 && curW < param.w && curH < param.h && innerColA * 4 + crs + BK + i < end_k){
                        // int inOffsetTmp = curH * inChannelOffset + curW * param.c + curC;
                        int inOffsetTmp = layout == 0 ? 
                                curH * inChannelOffset + curW * param.c + curC:
                                curC * inChannelOffset + curH * param.w + curW;
                        smeminput[write_flag * (BM+PAD) * BK + input_sts_addr + offset + i*(BM+PAD)] = param.input[inOffset + inOffsetTmp];
                    } else {
                        smeminput[write_flag * (BM+PAD) * BK + input_sts_addr + offset + i*(BM+PAD)] = 0.f;
                    }
                }
            }
        }
        // sts
        // for (int i = 0; i < 4; ++i)
        // {
        //     smemweight[write_flag * (BN+4) * 8 + weight_sts_addr + i * (BN+4)] = weight_ldg_reg[i];
        // }
        // for (int i = 0; i < 4; ++i)
        // {
        //     smeminput[write_flag * BM * 8 + input_sts_addr + i * BM] = input_ldg_reg[i];
        // }
        __syncthreads();
        write_flag ^= 1;
#pragma unroll
        for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx)
#pragma unroll
            for (uint i = 0; i < TM; ++i)
                input_frag[0][wSubRowIdx * TM + i] = smeminput[(load_flag ^ 1) * (BM+PAD) * BK +
                    input_lds_addr + wSubRowIdx * WSUBM + threadRowInWarp * TM + i];
#pragma unroll
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx)
#pragma unroll
            for (uint i = 0; i < TN; ++i)
                weight_frag[0][wSubColIdx * TN + i] = smemweight[(load_flag ^ 1) * (BN+PAD) * BK +
                    weight_lds_addr + wSubColIdx * WSUBN + threadColInWarp * TN + i];
// #pragma unroll
//         for (int i = 0; i < 4; ++i)
//         {
//             weight_frag[0][i] = smemweight[(load_flag ^ 1) * (BN+4) * 8 + weight_lds_addr + i];
//             weight_frag[0][i + 4] = smemweight[(load_flag ^ 1) * (BN+4) * 8 + weight_lds_addr + i + 16];
//         }
// #pragma unroll
//         for (int i = 0; i < 4; ++i)
//         {
//             input_frag[0][i] = smeminput[(load_flag ^ 1) * BM * 8 + input_lds_addr + i];
//             input_frag[0][i + 4] = smeminput[(load_flag ^ 1) * BM * 8 + input_lds_addr + i + 32];
//         }
#pragma unroll
        for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
#pragma unroll
            for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                // calculate per-thread results
#pragma unroll
                for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
#pragma unroll
                    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                        output_frag[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                                    (wSubColIdx * TN) + resIdxN] +=
                            input_frag[1][wSubRowIdx * TM + resIdxM] *
                            weight_frag[1][wSubColIdx * TN + resIdxN];
                    }
                }
            }
        }
// #pragma unroll
//         for (int i = 0; i < 8; ++i)
//         {
// #pragma unroll
//             for (int j = 0; j < 8; ++j)
//             {
//                 output_frag[i][j] += weight_frag[1][i] * input_frag[1][j];
//             }
//         }
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

    // constexpr uint OUTMITER = (TM * TN * WNITER * WMITER * NUM_THREADS) / (2 * BK * (BM + BN)) / OUTNITER;
    // const uint WMITER_TM_OUTMITER = WMITER * TM / OUTMITER;
    // const uint WNITER_TN_OUTNITER = WNITER * TN / OUTNITER;



//     // uint32_t bias_lds_addr = warp_id / 2 * 32;

// #pragma unroll
//     for (int i = 0; i < 2; ++i)
//     {
// #pragma unroll
//         for (int j = 0; j < 2; ++j)
//         {
//             __syncthreads();

// #pragma unroll
//             for (int subi = 0; subi < 4; ++subi)
//             {
// #pragma unroll
//                 for (int subj = 0; subj < 4; ++subj)
//                 {
//                     // output sts
//                     smemoutput[output_sts_addr + subi * 8 * 4 + subj] = output_frag[i * 4 + subi][j * 4 + subj];
//                 }
//             }
//             __syncthreads();

// #pragma unroll
//             for (int subk = 0; subk < 16; ++subk)
//             {
//                 int outOffset = z * param.k * param.Oh * param.Ow + (m_idx + i * 16 + subk) * param.Oh * param.Ow + n_idx + j * 32;
//                 if ((m_idx + i * 16 + subk) < param.k && (n_idx + j * 32) < param.Oh * param.Ow)
//                     param.output[outOffset] = smemoutput[output_lds_addr + subk * 32];
//             }
//         }
//     }
    const uint output_lds_addr = warp_id * WSUBM * WSUBN + lane_id;
    // const uint m_idx = by * BN + mma_tid_y * WN + threadColInWarp * WNITER_TN_OUTNITER;
    // const uint n_idx = bx * BM + mma_tid_x * WM + threadRowInWarp * WMITER_TM_OUTMITER;
    // const uint output_sts_addr = warp_id * WMITER_TM_OUTMITER * WNITER_TN_OUTNITER * WARPSIZE +
    //                     (threadRowInWarp * (WSUBN / TN)  + threadColInWarp) * WMITER_TM_OUTMITER * WNITER_TN_OUTNITER;
    const uint output_sts_addr = mma_tid_x * BN / WN * TM * TN * WARPSIZE + mma_tid_y * TM * TN * WARPSIZE +
                         threadColInWarp * TN * WSUBM + threadRowInWarp * TM;
    const uint m_idx = by * BN + mma_tid_y * WN;
    const uint n_idx = bx * BM + mma_tid_x * WM;

    // const int n = (ksplit > 0) ? n_idx / PQ : z;

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
#pragma unroll
            for (int subk = 0; subk < TM * TN; ++subk){
                const uint row =  m_idx + j * WSUBN + (lane_id + subk * WARPSIZE) / WSUBM;
                const uint gemm_i =  n_idx + i * WSUBM + (lane_id + subk * WARPSIZE) % WSUBM;
                const int n = (ksplit > 0) ? gemm_i / PQ : z;
                const int col = (ksplit > 0) ? gemm_i % PQ : gemm_i;

                if (n < param.n && row < param.k && col < param.Oh * param.Ow){
                //     int outOffset = z * param.n * param.k * param.Oh * param.Ow +  n * param.k * param.Oh * param.Ow  + (m_idx + i * 16 + subk) * param.Oh * param.Ow + (n_idx + j * 32);
                // if (n < param.n && (m_idx + i * 16 + subk) < param.k && (n_idx + j * 32) < param.Oh * param.Ow)
                //     param.interm[outOffset] = smemoutput[output_lds_addr + subk * 32];
                    if  constexpr (ksplit > 0){
                        const uint outOffset = z * param.n * param.k * param.Oh * param.Ow + n * param.k * param.Oh * param.Ow +
                                row * param.Oh * param.Ow + col;
                        param.interm[outOffset] = smemoutput[output_lds_addr + subk * WARPSIZE];
                    } else {
                        const uint outOffset = z * param.k * param.Oh * param.Ow +
                                row * param.Oh * param.Ow + col;
                        param.output[outOffset] = smemoutput[output_lds_addr + subk * WARPSIZE];
                    }
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
    const bool nchw = param.nchw;
    const int ksplit = param.ksplit;

    int outh = (h - r + 2 * p) / u + 1;
    int outw = (w - s + 2 * q) / v + 1;    

    const uint bm = 128;
    const uint bn = 128;
    const uint bk = 8;

    const uint NUM_THREADS = 256;
    
    const uint wn = 32;
    const uint wm = 64;
    const uint wniter = 2; // =1 answer is wrong
    const uint tn = 4;
    const uint tm = 4;
    const uint oniter = 2;

    const uint kslit = 8;

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

    static_assert(((wm / wmiter) % tm == 0 ) && ((wn / wniter) % tn == 0), "");

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

    // if(ksplit > 0){
    //     assert((c*r*s) % kslit == 0);
    // }

    // static_assert((tm * tn * wniter * wmiter * NUM_THREADS) % ( 2 * bk * (bm+bn)) == 0,
    //                 "total smem size (in # of floats) must be divisible by size of all regsister files holding outputs ");

    // static_assert(((tm * tn * wniter * wmiter * NUM_THREADS) / ( 2 * bk * (bm+bn))) % oniter == 0, "");

    const unsigned int nrows = n * k * outh * outw;
    // static_assert(2 * bk * (bm+bn) >= tm * tn * NUM_THREADS, "shared memory size must be larger than register file size");
    // int device = 0;
    // cudaDeviceProp prop;
    // cudaGetDeviceProperties(&prop, device);

    // printf("Shared memory per block: %zu bytes\n", prop.sharedMemPerBlock);
    // printf("Number of SMs: %zu \n", prop.multiProcessorCount);

    if(ksplit > 0){
        int blockx = ((n * outh * outw + bm-1) / bm); // blockx  number
        int blocky = (k + bn-1) / bn;             // blocky  number
        // int blockx = ((outh * outw + 63) / 64); // blockx  number
        // int blocky = (k + 63) / 64;             // blocky  number
        int blockz = kslit;                           // blockz  number
        // 合并threadx与thready
        int threadx = NUM_THREADS; // threadx number per block
        int thready = 1;   // thready number per block
        int threadz = 1;   // threadz number per block
        dim3 block(threadx, thready, threadz);
        dim3 grid(blockx, blocky, blockz);
        if(!nchw){ // NHWC layout
            if( c % 4 == 0 )
                implgemm<bm, bn, bk, wm, wn, wniter, tm, tn, NUM_THREADS, 0, true, true, kslit><<<grid, block>>>(param);
            else
                implgemm<bm, bn, bk, wm, wn, wniter, tm, tn, NUM_THREADS, 0, false, false, kslit><<<grid, block>>>(param);
        } else{ // NCHW layout
            implgemm<bm, bn, bk, wm, wn, wniter, tm, tn, NUM_THREADS, 1, false, false, kslit><<<grid, block>>>(param);
        }
        blockx = (nrows + 511) / 512;
        const dim3 block_nums(blockx, 1, 1);
        const dim3 block_dims(512, 1, 1);
        reduce_f32<<<block_nums, block_dims>>>(param.interm, param.output, nrows, kslit);
    }else{
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
        if(!nchw){ // NHWC layout
            if( c % 4 == 0 )
                implgemm<bm, bn, bk, wm, wn, wniter, tm, tn, NUM_THREADS, 0, true, true, 0><<<grid, block>>>(param);
            else
                implgemm<bm, bn, bk, wm, wn, wniter, tm, tn, NUM_THREADS, 0, false, false, 0><<<grid, block>>>(param);
        } else{ // NCHW layout
            implgemm<bm, bn, bk, wm, wn, wniter, tm, tn, NUM_THREADS, 1, false, false, 0><<<grid, block>>>(param);
        }
    }
    return cudaGetLastError();
}
