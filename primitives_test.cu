#include <stdio.h>
#include <cuda.h>
#include <iostream>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include "device_functions.h"
#include <thrust/scan.h>
#include <thrust/sort.h>
#include "CycleTimer.h"
#define SCAN_BLOCK_DIM 512
#define uint unsigned int
#include "exclusiveScan.cu_inl"
#include "cuPrintf.cu"
#include "scan.cu"
#include "book.h"
using namespace std;
extern float toBW(int bytes, float sec);


// This scan only work on small buffer, do not used on large array
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

/*
    choose the quilified tuples from the relation
    and get the cout of tuples of each block
*/
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

      for(int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if(threadIdx.x < offset) {
          // add a partial sum upstream to our own
          input[threadIdx.x] += input[threadIdx.x + offset];
        }

        // wait until all threads in the block have
        // updated their partial sums
        __syncthreads();
      }

      // thread 0 writes the final result
      if(threadIdx.x == 0) {
        result_size[blockIdx.x] = input[0];
      }
}

/*
    gather stage
*/
__global__ void coalesced(int N, int* result, int* result_size, int* histogram, int* out) {
	int threadIndex =  threadIdx.x;
	int partition = blockIdx.x *  blockDim.x;
    if( threadIndex < result_size[blockIdx.x] ) {
		out[histogram[blockIdx.x] + threadIndex] = result[partition + threadIndex];
	}
	__syncthreads();
}


/*
    This is a sample of how to use scanLargeArray
    from Nvidia SDK
*/
void primitive_scan(int N, int inData[], int outData[]) {
	int large_num = 39063;
    float tmp[large_num];
    float* large_in;
    float* large_out;
    double startTime;
    double endTime;
	cudaMalloc((void**) &large_in, sizeof(float) * large_num);
	cudaMalloc((void**) &large_out, sizeof(float) * large_num);
    //cudaMemset(large_in, 1, large_num * sizeof(float));
    for(int i = 0; i < large_num; i ++) {
        tmp[i] = 1.0;
    }
	cudaMemcpy(large_in, tmp, sizeof(float) * large_num, cudaMemcpyHostToDevice);
    startTime = CycleTimer::currentSeconds();
    preallocBlockSums(large_num);
    prescanArray(large_out, large_in, large_num);
    endTime = CycleTimer::currentSeconds();
   printf("time excution from large array scan %.3f ms\n", 1000.f * (endTime  - startTime));
   /* startTime = CycleTimer::currentSeconds();
    thrust::device_ptr<float> dev_ptr1(large_in);
    thrust::device_ptr<float> dev_ptr2(large_out);
    thrust::exclusive_scan(dev_ptr1, dev_ptr1 + large_num, dev_ptr2);
    endTime = CycleTimer::currentSeconds();
   printf("time excution from thrust scan %.3f ms\n",1000.f * (endTime  - startTime));*/
    cudaMemcpy(tmp, large_out, sizeof(float) * large_num, cudaMemcpyDeviceToHost);
    for(int i = 0; i < large_num; i ++) {
        printf("%f ", tmp[i]);
    }
    printf("\n");
    int y[] = {1, 2};
    printf("%d\n", y[(int)tmp[1]]);
    deallocBlockSums();
}


/*
    Implementation of SELECT operation
*/
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
	for(int i = 0 ; i < 10 ; i ++) {
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
    }
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

__device__ int get_index_to_check(int thread, int num_threads, int set_size, int offset) {

  // Integer division trick to round up
  return (((set_size + num_threads) / num_threads) * thread) + offset;
}

__device__ void search_lower(int search, int array_length,  int2 *arr, int *ret_val ) {
  const int num_threads = blockDim.x;
  const int thread = threadIdx.x;
  int set_size = array_length;
  while(set_size != 0){
    // Get the offset of the array, initially set to 0
    int offset = ret_val[1];

    // I think this is necessary in case a thread gets ahead, and resets offset before it's read
    // This isn't necessary for the unit tests to pass, but I still like it here
    __syncthreads();

    // Get the next index to check
    int index_to_check = get_index_to_check(thread, num_threads, set_size, offset);

    // If the index is outside the bounds of the array then lets not check it
    if (index_to_check < array_length){
      // If the next index is outside the bounds of the array, then set it to maximum array size
      int next_index_to_check = get_index_to_check(thread + 1, num_threads, set_size, offset);

      if (next_index_to_check >= array_length){
        next_index_to_check = array_length - 1;
      }
   /* if( search == 5 && blockIdx.x == 1) {
        cuPrintf("index to check arr[%d] = %d , arr[%d] = %d \n", index_to_check,arr[index_to_check].x, next_index_to_check, arr[next_index_to_check].x);
    }*/

      // If we're at the mid section of the array reset the offset to this index
      if (search > arr[index_to_check].x && (search <= arr[next_index_to_check].x)) {
        ret_val[1] = index_to_check;
      }
      else if (search == arr[index_to_check].x) {
        // Set the return var if we hit it
       /* if(blockIdx.x == 1 && search == 5) {
            cuPrintf("find it at %d %d\n", index_to_check, ret_val[0]);
        }*/
        atomicMin(&ret_val[0], index_to_check);
      }
    }

    // Since this is a p-ary search divide by our total threads to get the next set size
    set_size = set_size / num_threads;

    // Sync up so no threads jump ahead and get a bad offset
    __syncthreads();
  }
}

__device__ void search_upper(int search, int array_length,  int2 *arr, int *ret_val ) {
  const int num_threads = blockDim.x;
  const int thread = threadIdx.x;
  int set_size = array_length;
  while(set_size != 0){
    // Get the offset of the array, initially set to 0
    int offset = ret_val[1];

    // I think this is necessary in case a thread gets ahead, and resets offset before it's read
    // This isn't necessary for the unit tests to pass, but I still like it here
    __syncthreads();

    // Get the next index to check
    int index_to_check = get_index_to_check(thread, num_threads, set_size, offset);

    // If the index is outside the bounds of the array then lets not check it
    if (index_to_check < array_length){
      // If the next index is outside the bounds of the array, then set it to maximum array size
      int next_index_to_check = get_index_to_check(thread + 1, num_threads, set_size, offset);

      if (next_index_to_check >= array_length){
        next_index_to_check = array_length - 1;
      }
   /* if( search == 5 && blockIdx.x == 1) {
        cuPrintf("index to check arr[%d] = %d , arr[%d] = %d \n", index_to_check,arr[index_to_check].x, next_index_to_check, arr[next_index_to_check].x);
    }*/

      // If we're at the mid section of the array reset the offset to this index
      if (search > arr[index_to_check].x && (search <= arr[next_index_to_check].x)) {
        ret_val[1] = index_to_check;
      }
      else if (search == arr[index_to_check].x) {
        // Set the return var if we hit it
       /* if(blockIdx.x == 1 && search == 5) {
            cuPrintf("find it at %d %d\n", index_to_check, ret_val[0]);
        }*/
        atomicMax(&ret_val[0], index_to_check);
      }
    }

    // Since this is a p-ary search divide by our total threads to get the next set size
    set_size = set_size / num_threads;

    // Sync up so no threads jump ahead and get a bad offset
    __syncthreads();
  }
}
__global__ void p_ary_search(int search, int array_length,  int2 *arr, int *ret_val ) {

  const int num_threads = blockDim.x * gridDim.x;
  const int thread = blockIdx.x * blockDim.x + threadIdx.x;
  //ret_val[0] = -1;
  //ret_val[1] = 0;

  int set_size = array_length;


  while(set_size != 0){
    // Get the offset of the array, initially set to 0
    int offset = ret_val[1];

    // I think this is necessary in case a thread gets ahead, and resets offset before it's read
    // This isn't necessary for the unit tests to pass, but I still like it here
    __syncthreads();

    // Get the next index to check
    int index_to_check = get_index_to_check(thread, num_threads, set_size, offset);
    // If the index is outside the bounds of the array then lets not check it
    if (index_to_check < array_length){
      // If the next index is outside the bounds of the array, then set it to maximum array size
      int next_index_to_check = get_index_to_check(thread + 1, num_threads, set_size, offset);

      if (next_index_to_check >= array_length){
        next_index_to_check = array_length - 1;
      }

      // If we're at the mid section of the array reset the offset to this index
      if (search > arr[index_to_check].x && (search < arr[next_index_to_check].x)) {
        ret_val[1] = index_to_check;
      }
      else if (search == arr[index_to_check].x) {
        // Set the return var if we hit it
        ret_val[0] = index_to_check;
      }
    }

    // Since this is a p-ary search divide by our total threads to get the next set size
    set_size = set_size / num_threads;

    // Sync up so no threads jump ahead and get a bad offset
    __syncthreads();
  }
}

__global__ void pnary_partition(int2* rel_a, int2* rel_b, int* lower_array, int* upper_array, float* out_bound, int N, int M) {
	const int lower_bound = rel_a[blockIdx.x *  blockDim.x].x;
   	const int upper_bound = rel_a[(blockIdx.x + 1) * blockDim.x - 1].x;
    __shared__ int lower;
    __shared__ int upper;
    lower_array[2 * blockIdx.x] = M;
    lower_array[2 * blockIdx.x + 1] = 0;
    upper_array[2 * blockIdx.x] = -1;
    upper_array[2 * blockIdx.x + 1] = 0;
    __syncthreads();
    search_lower(lower_bound, M, rel_b, &lower_array[2 * blockIdx.x]);
    search_upper(upper_bound, M, rel_b, &upper_array[2 * blockIdx.x]);
    lower = lower_array[2 * blockIdx.x] < M? lower_array[2 * blockIdx.x]:lower_array[2 * blockIdx.x + 1];
    upper = upper_array[2 * blockIdx.x] >= 0? upper_array[2 * blockIdx.x]:upper_array[2 * blockIdx.x + 1];
    if( upper < lower) {
        upper = M - 1;
    }
    out_bound[blockIdx.x] = blockDim.x * ( upper - lower + 1);
    /*if(threadIndex == 0) {
    cuPrintf("lower_bound: %d ret: %d offset: %d\n", lower_bound, lower_array[2 * blockIdx.x], lower_array[2 * blockIdx.x + 1]);
    cuPrintf("upper_bound: %d ret: %d offset: %d\n", upper_bound, upper_array[2 * blockIdx.x], upper_array[2 * blockIdx.x + 1]);
    cuPrintf("num result tuples: %f\n", out_bound[blockIdx.x]);
    }*/
}

void
__global__ brute_join( int3* out, int2* rel_a, int2* rel_b, int num, int N, int M, float* out_bound, float* result_size, int* lower_array, int* upper_array ) {
    __shared__ int2 left[512];
    __shared__ int2 right[1024];
    __shared__ uint count[512];
    __shared__ uint index[512];
    __shared__ uint scratch[1024];
    int lower;
    int upper;
    int num_right;
    lower = lower_array[2 * blockIdx.x] < M? lower_array[2 * blockIdx.x]:lower_array[2 * blockIdx.x + 1];
    upper = upper_array[2 * blockIdx.x] >= 0? upper_array[2 * blockIdx.x]:upper_array[2 * blockIdx.x + 1];
    if( upper < lower) {
        upper = M - 1;
    }
    num_right = upper - lower + 1;
    int threadIndex =  threadIdx.x;
    int partition = blockIdx.x * blockDim.x;
    // counter for each thread
    count[threadIndex] = 0;
    // load two relation to the cache, make future access faster
    left[threadIndex] = rel_a[partition + threadIndex];
    for(int i = 0 ; i < num_right; i+= 512) {
        if(i + threadIndex < num_right) {
            //cuPrintf("%d\n",lower + i + threadIndex);
            right[i + threadIndex] = rel_b[lower + i + threadIndex];
        }
        __syncthreads();
   }
    for(int i = 0 ; i < num_right; i++ ) {
        if(left[threadIndex].x == right[i].x) {
            count[threadIndex] ++;
        }
    }
    __syncthreads();
    sharedMemExclusiveScan(threadIndex, count, index, scratch, SCAN_BLOCK_DIM);
    for(int i = 0 ; i < num_right; i++ ) {
        if(left[threadIndex].x == right[i].x) {
           // out[(int)out_bound[blockIdx.x] + index[threadIndex] + i].x = left[threadIndex].x;
           // out[(int)out_bound[blockIdx.x] + index[threadIndex] + i].y = left[threadIndex].y;
           // out[(int)out_bound[blockIdx.x] + index[threadIndex] + i].z = right[i].y;
            if( threadIndex == 0) {
                cuPrintf(" out index %d\n", (int)out_bound[blockIdx.x] + index[threadIndex] + i);
            }
        }
    }
}

/*
    Implementation of JOIN operationi
    rel_a: left relation
    rel_b: right relation
    N: size of rel_a
    M: size of rel_b
*/
struct compare_int2 {

    __host__ __device__
    bool operator()(int2 a, int2 b) {
        return a.x <= b.x;
    }
};

void primitive_join(int N, int M) {
    // prepare host buffers
    int min = 1;
    int max = 1024;
    int2* rel_a = new int2[N];
    int2* rel_b = new int2[M];
    int2* rel_a_seq = new int2[N];
    int2* rel_b_seq = new int2[M];
    for(int i = 0; i < N; i ++) {
        rel_a[i] = make_int2(min + (rand() % (int)(max - min + 1)), min + (rand() % (int)(max - min + 1)) );
	rel_a_seq[i] = make_int2(min + (rand() % (int)(max - min + 1)), min + (rand() % (int)(max - min + 1)) );
    }
    for(int i = 0; i < M; i ++) {
        rel_b[i] = make_int2(min + (rand() % (int)(max - min + 1)), min + (rand() % (int)(max - min + 1)) );
	rel_b_seq[i] = make_int2(min + (rand() % (int)(max - min + 1)), min + (rand() % (int)(max - min + 1)) );
    }
    thrust::sort(rel_a, rel_a + N, compare_int2());
    thrust::sort(rel_b, rel_b + M, compare_int2());

    thrust::sort(rel_a_seq, rel_a_seq + N, compare_int2());
    thrust::sort(rel_b_seq, rel_b_seq + M, compare_int2());

    // prepare device buffers
	const int threadPerBlock = 512;
	const int blocks = (N + threadPerBlock - 1) / threadPerBlock;
    printf("num blocks: %d\n", blocks);
    int2* dev_rel_a;
    int2* dev_rel_b;
    int2* dev_rel_a_seq;
    int2* dev_rel_b_seq;
    int* lower_array;
    int* upper_array;
    float* out_bound;
    //float* out_bound_scan;
    float* result_size;
    int3* out;
    int3* out_seq;
    cudaMalloc((void**) &out, sizeof(int) * M * N);
    cudaMalloc((void**) &out_seq, sizeof(int) * M * N);
    cudaMalloc((void**) &result_size, sizeof(float) * blocks);
    cudaMalloc((void**) &out_bound, sizeof(float) * blocks);
    cudaMalloc((void**) &lower_array, sizeof(int) * blocks * 2);
    cudaMalloc((void**) &upper_array, sizeof(int) * blocks * 2);
    cudaMalloc((void**) &dev_rel_a, sizeof(int2) * N);
    cudaMalloc((void**) &dev_rel_b, sizeof(int2) * M);
    cudaMalloc((void**) &dev_rel_a_seq, sizeof(int2) * N);
    cudaMalloc((void**) &dev_rel_b_seq, sizeof(int2) * M);
	cudaMemcpy(dev_rel_a, rel_a, sizeof(int2) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_rel_b, rel_b, sizeof(int2) * M, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_rel_a_seq, rel_a_seq, sizeof(int2) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_rel_b_seq, rel_b_seq, sizeof(int2) * M, cudaMemcpyHostToDevice);

    /*int counter = 0;
    for(int i = 0 ; i < N; i ++) {
        if( counter == threadPerBlock){
            printf("----------------------------------\n");
            counter = 0;
        }
            printf("a: [%d, %d]\n", rel_a[i].x, rel_a[i].y);
        counter++;
    }
    printf("----------------------------------\n");
    counter = 0;
    for(int i = 0 ; i < M; i ++) {
        if( counter == threadPerBlock){
            printf("----------------------------------\n");
            counter = 0;
        }
      printf("b: [%d, %d]\n", rel_b[i].x, rel_b[i].y);
      counter++;
    }*/
    cudaPrintfInit();

    sequential_join(dev_rel_a_seq, dev_rel_b_seq, N, out_seq);

    pnary_partition<<< blocks, threadPerBlock >>>(dev_rel_a, dev_rel_b, lower_array, upper_array ,out_bound, N, M);
    thrust::device_ptr<float> dev_ptr1(out_bound);
    thrust::exclusive_scan(dev_ptr1, dev_ptr1 + blocks, dev_ptr1);
    //prescanArray(out_bound, out_bound, blocks);
    //deallocBlockSums();
    brute_join<<< blocks, threadPerBlock >>>(out, dev_rel_a, dev_rel_b, M * N, N, M, out_bound, result_size, lower_array, upper_array);
    /*float* tmp_check = new float[blocks];
	cudaMemcpy(tmp_check, out_bound, sizeof(float) * blocks, cudaMemcpyDeviceToHost);
    for(int i = 0 ; i < blocks; i ++) {
        printf("### %d ",(int)tmp_check[i]);
    }
    printf("\n");*/

    cudaPrintfDisplay(stdout, true);
 	cudaPrintfEnd();
    cudaFree(dev_rel_a_seq);
    cudaFree(dev_rel_b_seq);
    cudaFree(dev_rel_a);
    cudaFree(dev_rel_b);
    cudaFree(lower_array);
    cudaFree(upper_array);
    cudaFree(out_bound);
    cudaFree(out_seq);
    //cudaFree(out_bound_scan);
    cudaFree(result_size);
    cudaFree(out);
  //  deallocBlockSums();
}

void sequential_join(int2* rel_a, int2* rel_b, int rel_a_size, int3* out) {
     int out_index = 0;
     for (int i = 0; i < rel_a_size; i++) {
     	out_index = find_matches(rel_a, rel_b, i, out, out_index);
     }
     return;
}

//brute force find matching tuples
int find_matches(int2* tuple, int2* rel, int a_index, int3* out, int current) {
    int new_index = current;
    int rel_index = 0;

    while (rel[rel_index].x <= tuple[a_index].x) {
    	  out[current].x = tuple[a_index].x;
	  out[current].y = tuple[a_index].y;
	  out[current].z = rel[rel_index].y;
	  rel_index++;
	  new_index++;
    }
    return new_index;
}

#define N   (1024*1024)
#define FULL_DATA_SIZE   (N*20)

__global__ void kernel( int *a, int *b, int *c ) {
 int idx = threadIdx.x + blockIdx.x * blockDim.x;
      if (idx < N) {
             int idx1 = (idx + 1) % 256;
             int idx2 = (idx + 2) % 256;
             float   as = (a[idx] + a[idx1] + a[idx2]) / 3.0f;
             float   bs = (b[idx] + b[idx1] + b[idx2]) / 3.0f;
             c[idx] = (as + bs) / 2;
        }
}

void  streamTest() {

  cudaDeviceProp  prop;
    int whichDevice;
    HANDLE_ERROR( cudaGetDevice( &whichDevice ) );
    HANDLE_ERROR( cudaGetDeviceProperties( &prop, whichDevice ) );
    if (!prop.deviceOverlap) {
        printf( "Device will not handle overlaps, so no speed up from streams\n" );
    }

    cudaEvent_t     start, stop;
    float           elapsedTime;

    cudaStream_t    stream0, stream1;
    int *host_a, *host_b, *host_c;
    int *dev_a0, *dev_b0, *dev_c0;
    int *dev_a1, *dev_b1, *dev_c1;

    // start the timers
    HANDLE_ERROR( cudaEventCreate( &start ) );
    HANDLE_ERROR( cudaEventCreate( &stop ) );

    // initialize the streams
    HANDLE_ERROR( cudaStreamCreate( &stream0 ) );
    HANDLE_ERROR( cudaStreamCreate( &stream1 ) );

    // allocate the memory on the GPU
    HANDLE_ERROR( cudaMalloc( (void**)&dev_a0,
                              N * sizeof(int) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_b0,
                              N * sizeof(int) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_c0,
                              N * sizeof(int) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_a1,
                              N * sizeof(int) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_b1,
                              N * sizeof(int) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_c1,
                              N * sizeof(int) ) );

    // allocate host locked memory, used to stream
    HANDLE_ERROR( cudaHostAlloc( (void**)&host_a,
                              FULL_DATA_SIZE * sizeof(int),
                              cudaHostAllocDefault ) );
    HANDLE_ERROR( cudaHostAlloc( (void**)&host_b,
                              FULL_DATA_SIZE * sizeof(int),
                              cudaHostAllocDefault ) );
    HANDLE_ERROR( cudaHostAlloc( (void**)&host_c,
                              FULL_DATA_SIZE * sizeof(int),
                              cudaHostAllocDefault ) );

    for (int i=0; i<FULL_DATA_SIZE; i++) {
        host_a[i] = rand();
        host_b[i] = rand();
    }

    HANDLE_ERROR( cudaEventRecord( start, 0 ) );
    // now loop over full data, in bite-sized chunks
    for (int i=0; i<FULL_DATA_SIZE; i+= N*2) {
        // enqueue copies of a in stream0 and stream1
        HANDLE_ERROR( cudaMemcpyAsync( dev_a0, host_a+i,
                                       N * sizeof(int),
                                       cudaMemcpyHostToDevice,
                                       stream0 ) );
        HANDLE_ERROR( cudaMemcpyAsync( dev_a1, host_a+i+N,
                                       N * sizeof(int),
                                       cudaMemcpyHostToDevice,
                                       stream1 ) );
        // enqueue copies of b in stream0 and stream1
        HANDLE_ERROR( cudaMemcpyAsync( dev_b0, host_b+i,
                                       N * sizeof(int),
                                       cudaMemcpyHostToDevice,
                                       stream0 ) );
        HANDLE_ERROR( cudaMemcpyAsync( dev_b1, host_b+i+N,
                                       N * sizeof(int),
                                       cudaMemcpyHostToDevice,
                                       stream1 ) );

        // enqueue kernels in stream0 and stream1
        kernel<<<N/256,256,0,stream0>>>( dev_a0, dev_b0, dev_c0 );
        kernel<<<N/256,256,0,stream1>>>( dev_a1, dev_b1, dev_c1 );

        // enqueue copies of c from device to locked memory
        HANDLE_ERROR( cudaMemcpyAsync( host_c+i, dev_c0,
                                       N * sizeof(int),
                                       cudaMemcpyDeviceToHost,
                                       stream0 ) );
        HANDLE_ERROR( cudaMemcpyAsync( host_c+i+N, dev_c1,
                                       N * sizeof(int),
                                       cudaMemcpyDeviceToHost,
                                       stream1 ) );
    }
    HANDLE_ERROR( cudaStreamSynchronize( stream0 ) );
    HANDLE_ERROR( cudaStreamSynchronize( stream1 ) );

    HANDLE_ERROR( cudaEventRecord( stop, 0 ) );

    HANDLE_ERROR( cudaEventSynchronize( stop ) );
    HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime,
                                        start, stop ) );
    printf( "Time taken:  %3.1f ms\n", elapsedTime );

    // cleanup the streams and memory
    HANDLE_ERROR( cudaFreeHost( host_a ) );
    HANDLE_ERROR( cudaFreeHost( host_b ) );
    HANDLE_ERROR( cudaFreeHost( host_c ) );
    HANDLE_ERROR( cudaFree( dev_a0 ) );
    HANDLE_ERROR( cudaFree( dev_b0 ) );
    HANDLE_ERROR( cudaFree( dev_c0 ) );
    HANDLE_ERROR( cudaFree( dev_a1 ) );
    HANDLE_ERROR( cudaFree( dev_b1 ) );
    HANDLE_ERROR( cudaFree( dev_c1 ) );
    HANDLE_ERROR( cudaStreamDestroy( stream0 ) );
    HANDLE_ERROR( cudaStreamDestroy( stream1 ) );
}

void sequentialTest() {
    cudaEvent_t     start, stop;
    float           elapsedTime;
    HANDLE_ERROR( cudaEventCreate( &start ) );
    HANDLE_ERROR( cudaEventCreate( &stop ) );
    int* host_a = new int[FULL_DATA_SIZE];
    int* host_b = new int[FULL_DATA_SIZE];
    int* host_c = new int[FULL_DATA_SIZE];
    int *dev_a, *dev_b, *dev_c;
    // allocate the memory on the GPU
    cudaMalloc( (void**)&dev_a, FULL_DATA_SIZE * sizeof(int));
    cudaMalloc( (void**)&dev_b, FULL_DATA_SIZE * sizeof(int));
    cudaMalloc( (void**)&dev_c, FULL_DATA_SIZE * sizeof(int));
    for (int i=0; i<FULL_DATA_SIZE; i++) {
        host_a[i] = rand();
        host_b[i] = rand();
    }
    HANDLE_ERROR( cudaEventRecord( start, 0 ) );
	cudaMemcpy(dev_a, host_a, sizeof(int) * FULL_DATA_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, host_b, sizeof(int) * FULL_DATA_SIZE, cudaMemcpyHostToDevice);
    kernel<<<FULL_DATA_SIZE/256,256,0>>>( dev_a, dev_b, dev_c);
    cudaMemcpyAsync( host_c, dev_c, FULL_DATA_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
    HANDLE_ERROR( cudaEventSynchronize( stop ) );
    HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime,
                                        start, stop ) );
    printf( "Time taken:  %3.1f ms\n", elapsedTime );
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
