#include <cstdint>

typedef struct __align__(16)
{
    float*   input;                                   //输入数据地址
    float*   weight;                                  //权值数据地址
    float*   bias;                                    //偏置值数据地址
    float*   output;                                  //输出数据地址
    float*   interm;                                  //输出数据地址
    unsigned int      n;                              //batch szie              
    unsigned int      c;                              //channel number          
    unsigned int      h;                              //数据高                  
    unsigned int      w;                              //数据宽                  
    unsigned int      k;                              //卷积核数量              
    unsigned int      r;                              //卷积核高                
    unsigned int      s;                              //卷积核宽                
    unsigned int      u;                              //卷积在高方向上的步长    
    unsigned int      v;                              //卷积在宽方向上的步长    
    unsigned int      p;                              //卷积在高方向上的补边    
    unsigned int      q;                              //卷积在宽方向上的补边    
    unsigned int      Oh;                             //卷积结果高             
    unsigned int      Ow;                             //卷积结果宽 
    bool              nchw;
    unsigned int      ksplit;                             //卷积结果宽 
    uint3 SC_fastdiv;
    uint3 OW_fastdiv;
    uint3 C_fastdiv;
    uint3 RS_fastdiv;
    uint3 S_fastdiv;
    unsigned int pad;
    unsigned int pad1;
    unsigned int pad2;
    // uint3 pad;
    // uint3 pad1;
}param_t;
// void launch_implgemm(param_t param);


// See https://gmplib.org/~tege/divcnst-pldi94.pdf figure 4.1.
// Precompute mp (m' in the paper) and L such that division
// can be computed using a multiply (high 32b of 64b result)
// and a shift:
//
// n/d = (mulhi(n, mp) + n) >> L;
static const uint3 init_fastdiv_values(uint32_t d) {
    

    // compute L = ceil(log2(d));
    uint32_t L = 0;
    while (L < 32 && (uint32_t{ 1 } << L) < d) {
        L++;
    }

    uint32_t mp = (uint32_t) ((uint64_t{ 1 } << 32) * ((uint64_t{ 1 } << L) - d) / d + 1);
    // pack divisor as well to reduce error surface
    return make_uint3(mp, L, d);
}

static __device__ __forceinline__ uint32_t fastdiv(uint32_t n, const uint3 fastdiv_values) {
    // expects fastdiv_values to contain <mp, L, divisor> in <x, y, z>
    // fastdiv_values.z is unused and optimized away by the compiler.
    // Compute high 32 bits of n * mp
    const uint32_t hi = __umulhi(n, fastdiv_values.x);
    // add n, apply bit shift
    return (hi + n) >> fastdiv_values.y;
}

static __device__ __forceinline__ uint32_t fastmodulo(uint32_t n, const uint3 fastdiv_values) {
    // expects  fastdiv_values to contain <mp, L, divisor> in <x, y, z> (see init_fastdiv_values)
    return n - fastdiv(n, fastdiv_values) * fastdiv_values.z;
}

cudaError_t launch_implgemm(param_t param);
