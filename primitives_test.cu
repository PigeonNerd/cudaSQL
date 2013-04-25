#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include "device_functions.h"
#include <iostream>

//#include "CycleTimer.h"
#define SCAN_BLOCK_DIM 256
#define uint unsigned int
#include "exclusiveScan.cu_inl"
#include "cuPrintf.cu"

using namespace std;

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
		 atomicAdd(result_size + blockIdx.x, 1);
    	 result[partition + output[threadIndex]] = tuples[partition + threadIndex];
 	}
     __syncthreads();
}

/*
	this is a temporary implementation
*/
__global__ void tmpScan(int numBlocks, int* result_size, int* histogram) {
	__shared__ uint input[SCAN_BLOCK_DIM];
	__shared__ uint output[SCAN_BLOCK_DIM];
	__shared__ uint scratch[2 * SCAN_BLOCK_DIM];
	int threadIndex =  threadIdx.x;
	input[threadIndex] = 0;
	if( threadIndex < numBlocks) {
		input[threadIndex] = result_size[threadIndex];
	}
	__syncthreads();
	 sharedMemExclusiveScan(threadIndex, input, output, scratch, SCAN_BLOCK_DIM);
	 if(input[threadIndex]){
	 	histogram[threadIndex] = output[threadIndex];
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
	int* device_in;
	int* device_result;
	cudaMalloc((void**) &device_in, sizeof(int) * 8);
	cudaMalloc((void**) &device_result, sizeof(int) * 8);
	
	int billy [8] = { 16, 2, 77, 40, 12071 , 222, 333, 444};
	cudaMemcpy(device_in, billy, sizeof(int) * 8, cudaMemcpyHostToDevice);
	cudaPrintfInit();
	prescan<<<2, 2, 16 * sizeof(int) >>>(device_result, device_in, 8);
	cudaThreadSynchronize();
	cudaMemcpy(outData, device_result, sizeof(int) * 8, cudaMemcpyDeviceToHost);

    for(int i = 0; i < 8 ; i ++) {
        cout<< outData[i] << ", ";
    }
    cout << endl;
	cudaPrintfDisplay(stdout, true);
 	cudaPrintfEnd();
}

void 
primitive_select(int N, int inData[], int outData[]) {
	const int threadPerBlock = 256;
	const int blocks = (N + threadPerBlock - 1) / threadPerBlock;
	const int blocksOfReulstSize = ( blocks + threadPerBlock - 1) / threadPerBlock;
    printf("Num of tuples %d\n", N);
	printf("Num of blocks %d\n", blocks);
	printf("Num of blocks for result size %d\n", blocksOfReulstSize);
    int* device_in;
	int* device_result;
	int* result_size;
	int* histogram;
	int* out;

	int* tmp = (int*)calloc(N, sizeof(int));
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
	primitive_select_kernel<<<blocks, threadPerBlock>>>(N, device_in, device_result, result_size);
	cudaThreadSynchronize();
	prescan<<<blocksOfReulstSize, threadPerBlock, threadPerBlock * 2 * sizeof(int)>>>(histogram, result_size, blocks);
    
	coalesced<<<blocks, threadPerBlock>>>(N, device_result, result_size, histogram, out);
	cudaPrintfDisplay(stdout, true);
 	cudaPrintfEnd();
	cudaMemcpy(outData, out, sizeof(int) * N, cudaMemcpyDeviceToHost);

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
