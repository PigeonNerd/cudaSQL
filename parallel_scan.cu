#ifndef _PARALLEL_SCAN_CU
#define _PARALLEL_SCAN_CU

#define b2(x)       (   (x) | (   (x) >> 1) )
#define b4(x)       ( b2(x) | ( b2(x) >> 2) )
#define b8(x)       ( b4(x) | ( b4(x) >> 4) )
#define b16(x)      ( b8(x) | ( b8(x) >> 8) )
#define b32(x)      (b16(x) | (b16(x) >>16) )
#define next_power_of_2(x)      (b32(x-1) + 1)

#define ARR_SIZE  2048 
#define ARR_SIZE_PAD    next_power_of_2(ARR_SIZE)
#define TILE_SIZE  1024
#define SUM_SIZE        ARR_SIZE_PAD/TILE_SIZE

__global__ void scan_up(float *in, float *out) {
    // float4 element ?
    // unrool ?

    __shared__ float temp[TILE_SIZE];
    int thid = threadIdx.x;
    int offset = 1;
    int offset_arr = blockIdx.x * TILE_SIZE;

    // load data into shared memory
    temp[2 * thid]     = in[offset_arr + 2 * thid];
    temp[2 * thid + 1] = in[offset_arr + 2 * thid + 1];
    __syncthreads();

    // build sum
    for (int d = TILE_SIZE>>1; d > 0; d >>= 1) {
        if (thid < d) {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
        __syncthreads();
    }

    // write result to device memory
    out[offset_arr + 2 * thid]     = temp[2 * thid];
    out[offset_arr + 2 * thid + 1] = temp[2 * thid + 1];
}


__global__ void scan_sum(float *in, float *out) {
    __shared__ float temp[SUM_SIZE];
    int thid = threadIdx.x;
    int offset = 1;

    // load sums into shared memory (check boundry)
    temp[2 * thid]     = out[(2 * thid + 1) * TILE_SIZE - 1];
    temp[2 * thid + 1] = out[(2 * thid + 2) * TILE_SIZE - 1];
    __syncthreads();

    // build sum
    for (int d = SUM_SIZE>>1; d > 0; d >>= 1) {
        if (thid < d) {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
        __syncthreads();
    }

    // set last element to 0
    if (thid == 0) {
        temp[SUM_SIZE - 1] = 0;
    }

    // traverse down
    for (int d = 1; d < SUM_SIZE; d *= 2) {
        offset >>= 1;
        if (thid < d) {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
        __syncthreads();
    }

    // distribute scan result (check boundry)
    out[(2 * thid + 1) * TILE_SIZE - 1] = temp[2 * thid];
    out[(2 * thid + 2) * TILE_SIZE - 1] = temp[2 * thid + 1];
}

__global__ void scan_down(float *in, float *out) {
    __shared__ float temp[TILE_SIZE];
    int thid= threadIdx.x;
    int offset = TILE_SIZE;
    int offset_arr = blockIdx.x * TILE_SIZE;

    // load data into shared memory
    temp[2 * thid]     = out[offset_arr + 2 * thid];
    temp[2 * thid + 1] = out[offset_arr + 2 * thid + 1];
    __syncthreads();

    // traverse down tree & build scan
    for (int d = 1; d < TILE_SIZE; d *= 2) {
        offset >>= 1;
        if (thid < d) {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
        __syncthreads();
    }

    // write result to device memory
    out[offset_arr + 2 * thid]     = temp[2 * thid];
    out[offset_arr + 2 * thid + 1] = temp[2 * thid + 1];
}


#endif
