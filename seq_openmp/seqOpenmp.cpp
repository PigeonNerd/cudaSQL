#include <stdlib.h>
#include <stdio.h>
#include <cstdlib>
#include <string>
#include <getopt.h>
#include "CycleTimer.h"
#include "seqOpenmp.h"
#include <omp.h>

#define magnitude 3

void seq_openmp() {

  int base = 1;

   for (int i = 0; i < magnitude; i++) {
    base <<= 1;
  }

  int NUM_TUPLES = base;
  int* relation = new int[NUM_TUPLES];
  int* result = new int[NUM_TUPLES];
  int* openmp_result = new int[NUM_TUPLES];
  int* openmp_index_buffer = new int[NUM_TUPLES];
  int* openmp_index_sum = new int[NUM_TUPLES];

  for (int i = 0; i < NUM_TUPLES; i++) {
    //    relation[i] = rand() % 1000 + 1;
    relation[i] = 2;
    result[i] = 0;
    openmp_index_buffer[i] = 0;
  }

  sequential_select(NUM_TUPLES, relation, result);

  double startTime = CycleTimer::currentSeconds();
  openmp_select(NUM_TUPLES, relation, openmp_index_buffer);
  openmp_select_tuples(NUM_TUPLES, openmp_index_buffer, relation, openmp_result, openmp_index_sum);
  double endTime = CycleTimer::currentSeconds();
  double overallDuration = endTime - startTime;

  printf("Sequential OpenMP overall: %.3f ms\t\t[%.3f GB/s]\n", 1000.f * overallDuration, toBW(NUM_TUPLES * sizeof(int) * 2, overallDuration));

  validate(NUM_TUPLES, openmp_result, result);
}

void sequential_select(int N, int inData[], int outData[]) {
  int counter = 0;
  for(int i = 0; i < N; i ++) {
    if(inData[i] % 2 == 0) {
      outData[counter] = inData[i];
      counter++;
    }
  }
}

void openmp_select_tuples(int N, int bufferData[] , int inData[], int outData[], int sumData[]) {
  /*PREFIX SUM/SCAN ON BUFFERDATA
    Then, populate outData with inData values
   */

  memcpy(sumData, bufferData, sizeof(int) * N);
  openmp_prefixsum(N, sumData);

#pragma omp parallel for
  for (int i = 0; i < N; i++) {
    printf("sumData[%d] = %d\n", i, sumData[i]);
    printf("bufferData[%d] = %d\n", i, bufferData[i]);
    if (bufferData[i]) {
      int index = sumData[i];
      printf("inData[%d] = %d", i, inData[i]);
      outData[index] = inData[index];
    }
    printf("\n\n");
  }
}

void openmp_prefixsum(int N, int sumData[]){

  /*
  int j, tid;

#pragma omp parallel shared(N, sumData, bufferData) private(j, tid)
  {
    tid = omp_get_thread_num();
    for (j = 1; j < N; j = 2*j)
      {
	if (tid >= j) {
	  sumData[tid] = bufferData[tid] + bufferData[tid-j];
	  #pragma omp barrier

	  bufferData[tid] = sumData[tid];
	  #pragma omp barrier
	}
      }
  }
  */

  sumData[0] = 0;
  for (int i = 1; i < N; i++) {
    sumData[i] += sumData[i-1];
  }

  printf("\n\n");

  /*  int num_threads, work;
  int *partial, *temp;
  int i, mynum, last;

  for (i = 0; i < N; i++) {
#pragma omp parallel default(none) private(i, mynum, last) shared(partial, temp, num_threads, work, N, sumData)
    {
#pragma omp single
      {
	num_threads = omp_get_num_threads();
	if(!(partial = (int *) malloc (sizeof (int) * num_threads))) exit(-1);
	if(!(temp = (int *) malloc (sizeof (int) * num_threads))) exit(-1);
	work = N / num_threads + 1;
      }

      mynum = omp_get_thread_num();
      for (i = work * mynum; i < work * mynum + work && i < N; i++) {
	sumData[i] += sumData[i-1];
      }
      partial[mynum] = sumData[i];

#pragma omp barrier
      for (i = 1; i < num_threads; i <<= 1) {
	if (mynum >= i)
	  temp[mynum] = partial[mynum] + partial[mynum - i];
#pragma omp barrier
#pragma omp single
	memcpy(partial + 1, temp+1, sizeof(int) * (num_threads - 1));
      }
      for (i = work * mynum; i < (last = work*mynum + work < N ? work *mynum + work : N); i++)
	sumData[i] += partial[mynum] - sumData[last-1];
    }
    return;
  }
  */
}

void openmp_select(int N, int inData[], int outData[]) {
#pragma omp parallel for schedule(dynamic, 1)
  for(int i = 0; i < N; i ++) {
      if(inData[i] % 2 == 0){
	outData[i] = 1;
      }
    }
}

bool validate(int N, int* openmp, int* target) {
   for(int i = 0 ; i < N; i ++) {
    if(openmp[i] != target[i]) {
      printf("ERROR: Result tuple %d not match, Expected: %d, Actual: %d\n",i, openmp[i] \
	     ,
	     target[i]);
    }
  }
  printf("SELECT PASS\n");
  return true;
}

float toBW(int bytes, float sec) {
  return static_cast<float>(bytes) / (1024. * 1024. * 1024.) / sec;
}
