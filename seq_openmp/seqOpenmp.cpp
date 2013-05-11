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
  double startTime;
  double endTime;
  double overallDuration;

  int NUM_TUPLES = base;
  int* relation = new int[NUM_TUPLES];
  int* result = new int[NUM_TUPLES];
  int* openmp_result = new int[NUM_TUPLES];
  for (int i = 0; i < NUM_TUPLES; i++) {
    //relation[i] = rand() % 1000 + 1;
    relation[i] = 1;
    result[i] = 0;
  }


  startTime = CycleTimer::currentSeconds();
  sequential_select(NUM_TUPLES, relation, result);
  endTime = CycleTimer::currentSeconds();
  overallDuration = endTime - startTime;
  openmp_select(NUM_TUPLES, relation, openmp_result);

  validate(NUM_TUPLES, openmp_result, result);
  printf("Sequential overall: %.3f ms\t\t[%.3f GB/s]\n", 1000.f * overallDuration, toBW(NUM_TUPLES * sizeof(int) * 2, overallDuration));
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


void prefix_sum(int n, int x[], int t[]) {
    int j;
    int tid;
    #pragma omp parallel shared(n,x,t) private(j,tid) num_threads(n)
    {
       tid = omp_get_thread_num();
       for (j = 1; j < n; j = 2*j) {
            if (tid >= j)
                   t[tid] = x[tid] + x[tid - j];
             #pragma omp barrier
             x[tid] = t[tid];
             #pragma omp barrier
         } 
    }
}

void openmp_select(int N, int inData[], int outData[]) {
    int input[N];
    int output[N];

 #pragma omp parallel for schedule(dynamic, 1)
   for(int i = 0; i < N; i ++) {
       //input[i] = 0;
     //if(inData[i] % 2 == 0){
 	     input[i] = 1;
       //}
   }

   prefix_sum(N, input, output);
   for(int i = 0 ; i < N; i ++) {
        printf("%d ", input[i]);
   }
   printf("\n");

}

bool validate(int N, int* openmp, int* target) {
   for(int i = 0 ; i < N; i ++) {
    if(openmp[i] != target[i]) {
      printf("ERROR: Result tuple %d not match, Expected: %d, Actual: %d\n",i, openmp[i], target[i]);
    }
  }
  printf("SELECT PASS\n");
  return true;
}

float toBW(int bytes, float sec) {
  return static_cast<float>(bytes) / (1024. * 1024. * 1024.) / sec;
}
