#include <stdio.h>
#include <cuda.h>
#include <iostream>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include "device_functions.h"
#include <thrust/scan.h>
#include "CycleTimer.h"
#define SCAN_BLOCK_DIM 512 
#define uint unsigned int
#include "exclusiveScan.cu_inl"
#include "cuPrintf.cu"
#include "scan.cu"

using namespace std;
extern float toBW(int bytes, float sec);

__global__ void prescan(int *g_odata, int *g_idata, int n){

    extern __shared__ int temp[];// allocated on invocation
    int thid = threadIdx.x;
    int offset = 1;
    temp[2*thid] = g_idata[2*thid]; // load input into shared memory
    temp[2*thid+1] = g_idata[2*thid+1];
    for (int d = n>>1; d > 0; d >>= 1){ // build sum in place up the tree
        __syncthreads();
        if (thid < d){

            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    if (thid == 0) { temp[n - 1] = 0; } // clear the last element
    for (int d = 1; d < n; d *= 2){ // traverse down tree & build scan
        offset >>= 1;
        __syncthreads();
        if (thid < d){
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;
            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();
    g_odata[2*thid] = temp[2*thid]; // write results to device memory
    g_odata[2*thid+1] = temp[2*thid+1];
}

__global__ void
primitive_select_kernel(int N, int* tuples, int* result, int* result_size) {

	__shared__ uint input[SCAN_BLOCK_DIM];
	__shared__ uint output[SCAN_BLOCK_DIM];
	__shared__ uint scratch[2 * SCAN_BLOCK_DIM];

	int threadIndex =  threadIdx.x;
	int partition = blockIdx.x *  blockDim.x;
	//cuPrintf("%d\n", threadIndex);
	input[threadIndex] = 0;
	output[threadIndex] = 0;
 	if ( partition + threadIndex < N ) {
		input[threadIndex] = tuples[partition + threadIndex] % 2 == 0? 1 : 0;
 	}
	 __syncthreads();
	 sharedMemExclusiveScan(threadIndex, input, output, scratch, SCAN_BLOCK_DIM);
	if(input[threadIndex]){
		 //atomicAdd(result_size + blockIdx.x, 1);
    	 result[partition + output[threadIndex]] = tuples[partition + threadIndex];
 	}
     //__syncthreads();
    // reduction phase
    extern __shared__ int sdata;
    unsigned int i = partition + threadIndex;
    int x = 0;
    if(i < n)
      {
        x = input[i];
      }
    sdata[threadIdx.x] = x;
    __syncthreads();
    for(int offset = blockDim.x / 2;
      offset > 0;
      offset >>= 1)
  {
    if(threadIdx.x < offset)
    {
      // add a partial sum upstream to our own
      sdata[threadIdx.x] += sdata[threadIdx.x + offset];
    }

    // wait until all threads in the block have
    // updated their partial sums
    __syncthreads();
  }

  // thread 0 writes the final result
  if(threadIdx.x == 0)
  {
    result_size[blockIdx.x] = sdata[0];
  }
}

__global__ void coalesced(int N, int* result, int* result_size, int* histogram, int* out) {
	int threadIndex =  threadIdx.x;
	int partition = blockIdx.x *  blockDim.x;
    if( threadIndex < result_size[blockIdx.x] ) {
		out[histogram[blockIdx.x] + threadIndex] = result[partition + threadIndex];
	}
	__syncthreads();
}

void
primitive_scan(int N, int inData[], int outData[]) {
	int large_num = 512;
    float tmp[large_num];
    float* large_in;
    float* large_out;

	cudaMalloc((void**) &large_in, sizeof(float) * large_num);
	cudaMalloc((void**) &large_out, sizeof(float) * large_num);
    
    for(int i = 0; i < large_num; i ++) {
        tmp[i] = 1.0;
    }
	cudaMemcpy(large_in, tmp, sizeof(float) * large_num, cudaMemcpyHostToDevice);
    preallocBlockSums(large_num);
    prescanArray(large_out, large_in, large_num);
	cudaMemcpy(tmp, large_out, sizeof(float) * large_num, cudaMemcpyDeviceToHost);
    for(int i = 0; i < large_num; i ++) {
        printf("%f ", tmp[i]);
    }
    printf("\n");
}

void 
primitive_select(int N, int inData[], int outData[]) {
	const int threadPerBlock = 512;
	const int blocks = (N + threadPerBlock - 1) / threadPerBlock;
	const int blocksOfReulstSize = ( blocks + threadPerBlock - 1) / threadPerBlock;
    int totalBytes = N * sizeof(int) * 2;
    printf("Num of tuples %d\n", N);
	printf("Num of blocks %d\n", blocks);
	printf("Num of blocks for result size %d\n", blocksOfReulstSize);
    
    int* device_in;
	int* device_result;
	int* result_size;
	int* histogram;
	int* out;
	int* tmp = (int*)calloc(N, sizeof(int));
    double startTime = CycleTimer::currentSeconds();
	cudaMalloc((void**) &device_in, sizeof(int) * N);
	cudaMalloc((void**) &device_result, sizeof(int) * N);
	cudaMalloc((void**) &out, sizeof(int) * N);
	cudaMalloc((void**) &result_size, sizeof(int) * blocks);
	cudaMalloc((void**) &histogram, sizeof(int) * blocks);

	cudaMemcpy(device_in, inData, sizeof(int) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(device_result, tmp, sizeof(int) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(out, tmp, sizeof(int) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(result_size, tmp, sizeof(int) * blocks, cudaMemcpyHostToDevice);
    cudaPrintfInit();
    double startTime_inner = CycleTimer::currentSeconds();	
	primitive_select_kernel<<<blocks, threadPerBlock>>>(N, device_in, device_result, result_size);

   // int test_result_size[blocks];
   // cudaMemcpy(test_result_size, result_size, sizeof(int)*blocks, cudaMemcpyDeviceToHost);
   // for(int i = 0 ; i < blocks ; i ++) {
   //     printf("%d, ", test_result_size[i]);
   // }
   // printf("\n");
	cudaThreadSynchronize();
	//prescan<<<blocksOfReulstSize, threadPerBlock, blocks * threadPerBlock * 2 * sizeof(int)>>>(histogram, result_size, blocks);
    
    thrust::device_ptr<int> dev_ptr1(result_size);
    thrust::device_ptr<int> dev_ptr2(histogram);
    thrust::exclusive_scan(dev_ptr1, dev_ptr1 + blocks, dev_ptr2);
   // int test_histgram[blocks];
   // cudaMemcpy(test_histgram, histogram, sizeof(int)*blocks, cudaMemcpyDeviceToHost);
   // for(int i = 0 ; i < blocks; i ++) {
   //     printf("%d, ", test_histgram[i]);
   // }
   // printf("\n");
	coalesced<<<blocks, threadPerBlock>>>(N, device_result, result_size, histogram, out);
    double endTime_inner = CycleTimer::currentSeconds();
    cudaPrintfDisplay(stdout, true);
 	cudaPrintfEnd();
    cudaMemcpy(outData, out, sizeof(int) * N, cudaMemcpyDeviceToHost);
    double endTime = CycleTimer::currentSeconds();
    
    double overallDuration = endTime - startTime;
    double kernelDuration = endTime_inner - startTime_inner;
    
    printf("CUDA overall: %.3f ms\t\t[%.3f GB/s]\n", 1000.f * overallDuration, toBW(totalBytes, overallDuration));
    printf("CUDA execution time for kernel: %.3f ms\t\t[%.3f GB/s]\n", 1000.f*kernelDuration, toBW(totalBytes, kernelDuration));
    cudaFree(device_in);
    cudaFree(device_result);
    cudaFree(out);
    cudaFree(result_size);
    cudaFree(histogram);
}	

void
printCudaInfo() {
    // for fun, just print out some stats on the machine

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
}
