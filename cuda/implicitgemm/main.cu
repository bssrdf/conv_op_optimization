#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>
// #include <cuda_ext.h>
#include "verify.h"

#include "conv2d.h"

#define OPENCNN_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"Error occurred: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);  
   }
}

int main(int argc, char **argv)
{
    unsigned int n = atoi(argv[1]);
    unsigned int c = atoi(argv[2]);
    unsigned int h = atoi(argv[3]);
    unsigned int w = atoi(argv[4]);
    unsigned int k = atoi(argv[5]);
    unsigned int r = atoi(argv[6]);
    unsigned int s = atoi(argv[7]);
    unsigned int u = atoi(argv[8]);
    unsigned int v = atoi(argv[9]);
    unsigned int p = atoi(argv[10]);
    unsigned int q = atoi(argv[11]);
    unsigned int nchw = atoi(argv[12]);

    int outh = (h - r + 2 * p) / u + 1;
    int outw = (w - s + 2 * q) / v + 1;
    double M = k;
    double N = n * outh * outw;
    double K = c * r * s;
    double temp = n * outh * outw * 1e-9f;
    double flopsPerConv = temp * M * K * 2.0;
    float *input = (float *)malloc(n * c * h * w * sizeof(float));
    float *weight = (float *)malloc(k * c * r * s * sizeof(float));
    float *bias = (float *)malloc(k * sizeof(float));
    float *output = (float *)malloc(n * k * outh * outw * sizeof(float));
    float *output_host = (float *)malloc(n * k * outh * outw * sizeof(float));

    float *input_device, *weight_device, *bias_device, *output_device;
    cudaMalloc((void **)&input_device, n * c * h * w * sizeof(float));
    cudaMalloc((void **)&weight_device, k * c * r * s * sizeof(float));
    cudaMalloc((void **)&bias_device, k * sizeof(float));
    cudaMalloc((void **)&output_device, n * k * outh * outw * sizeof(float));

    for (int i = 0; i < n * c * h * w; i++)
    {
        input[i] = (rand() % 255) / 255.0;
    }

    // for (int i = 0; i < k * c * r * s; i++)
    // {
    //     // weight[i] = (rand() % 255) / 255.0;
    //     weight[i] =  i / 1000.0;
    // }
    for(int j= 0; j < k; j++){
    for(int C= 0; C < c; C++){
    for (int i = 0; i < r * s; i++){
        // weight[i] = (rand() % 255) / 255.0;
        weight[j*r*s*c + C*r*s + i] = j * 10 + C/100.0 + i / 1000.0;
    }
    }
    }

    // for(int j= 0; j < k; j++)
    // for(int C= 0; C < 24; C++)
    //     printf("%d, %f\n", C, weight[C*r*s + 0]);
    

    for (int i = 0; i < k; i++)
    {
        // bias[i] = (rand() % 255) / 255.0;
        bias[i] = 0.f;
    }

    // for (int i = 0; i < n * k * outh * outw; i++)
    // {
    //     output[i] = 0.0;
    //     output_host[i] = 0.0;
    // }

    cudaMemcpy(input_device, input, n * c * h * w * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(weight_device, weight, k * c * r * s * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(bias_device, bias, k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(output_device, output, n * k * outh * outw * sizeof(float), cudaMemcpyHostToDevice);

    //Convolution parameter

    param_t param;

    param.input = input_device;
    uintptr_t base = reinterpret_cast<uintptr_t>(param.input);
    printf("param.input base = %p  (mod16 = %zu)\n", (void*)base, base % 16);
    assert((base % 16) == 0 && "param.input base is not 16-byte aligned");
    param.weight = weight_device;
    param.bias = bias_device;
    param.output = output_device;
    param.n = n;
    param.c = c;
    param.h = h;
    param.w = w;
    param.k = k;
    param.r = r;
    param.s = s;
    param.u = u;
    param.v = v;
    param.p = p;
    param.q = q;
    param.Oh = outh;
    param.Ow = outw;
    param.nchw = (nchw == 1) ? true : false;

    printf("launch implgemm, n:%d, c:%d, h:%d, w:%d, k:%d, r:%d, s:%d, u:%d, v:%d, p:%d, q:%d, outh:%d, outw:%d\n",
           n, c, h, w, k, r, s, u, v, p, q, outh, outw);
    /********************************** step 2****************************/


    /*******************************warm up and get result************************************/
    OPENCNN_CALL(launch_implgemm(param));

    cudaMemcpy(output_host, output_device, n * k * outh * outw * sizeof(float), cudaMemcpyDeviceToHost);

    /*******************************cost time test************************************/
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    float time_elapsed = 0.0;

    int iternum = 20;
    for (int i = 0; i < iternum; i++)
    {
        OPENCNN_CALL(launch_implgemm(param));
    }
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_elapsed, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("===================start verfiy===================\n");
    if(param.nchw)
        direct_conv2dcpu(input, weight, bias, output, n, c, h, w, k, r, s, u, v, p, q);
    else
        direct_conv2dcpu_nhwc(input, weight, bias, output, n, c, h, w, k, r, s, u, v, p, q);

    int error = 0;
    for (int i = 0; i < n * k * outh * outw; i++)
    {
        // printf(" postion:%d, gpuvalue:%f, cpuvalue:%f\n", i, output_host[i], output[i]);
        if (abs(output_host[i] - output[i]) > getPrecision(output[i]))
        {
            printf("error, postion:%d, gpuvalue:%f, cpuvalue:%f\n", i, output_host[i], output[i]);
            error++;
            break;
        }
            
    }
    printf("================finish,error:%d=========================\n", error);

    float timePerConv = time_elapsed / iternum;
    double gflops = flopsPerConv / (timePerConv / 1000.0f);
    printf("%2d %2d %2d %2d %d %d %2d\n", n, h, w, c, r, s, k);
    printf("time: %f ms\n", timePerConv);
    printf("Performance :%f GFlops\n",  gflops);
    
    cudaFree(input_device);
    cudaFree(weight_device);
    cudaFree(output_device);

    free(input);
    free(weight);
    free(output);
    free(output_host);

    return 0;
}