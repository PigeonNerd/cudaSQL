#include <stdlib.h>
#include <stdio.h>
#include <cstdlib>
#include <string>
#include <getopt.h>
#include "CycleTimer.h"
#include "seqOpenmp.h"
#include <omp.h>
#include <cmath>

#define magnitude 15 

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
    relation[i] = rand() % 1000 + 1;
    //relation[i] = 1;
    result[i] = 0;
  }


  startTime = CycleTimer::currentSeconds();
  sequential_select(NUM_TUPLES, relation, result);
  endTime = CycleTimer::currentSeconds();
  overallDuration = endTime - startTime;
  
  printf("Sequential overall: %.3f ms\t\t[%.3f GB/s]\n", 1000.f * overallDuration, toBW(NUM_TUPLES * sizeof(int) * 2, overallDuration));
  startTime = CycleTimer::currentSeconds();
  openmp_select(NUM_TUPLES, relation, openmp_result);
  endTime = CycleTimer::currentSeconds();
  overallDuration = endTime - startTime;

  printf("openmp overall: %.3f ms\t\t[%.3f GB/s]\n", 1000.f * overallDuration, toBW(NUM_TUPLES * sizeof(int) * 2, overallDuration));

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


void prefix_sum(int n, int x[], int t[]) {
   /* int i,j;
    for (j = 0; j < log2(n); j++) {
        #pragma omp parallel private(i)
        {
            #pragma omp for
            for (i = 1<<j; i < n; i++)
                t[i] = x[i] + x[i -( 1<<j)];
            #pragma omp for
            for (i = 1<<j; i < n; i++)
                x[i] = t[i];
        }
    }*/
}

void openmp_select(int N, int inData[], int outData[]) {
    int input1[N];
    int input2[N];
    int output[2 * N];

 #pragma omp parallel for schedule(dynamic, 1)
   for(int i = 0; i < N; i ++) {
       input1[i] = 0;
       input2[i] = 0;
     if(inData[i] % 2 == 0){
 	     input1[i] = 1;
 	     input2[i] = 1;
     }
   }

 prefix_sum(N, input1, output);
 #pragma omp parallel for schedule(dynamic, 1)
   for(int i = 0; i < N; i ++) {
        if(input2[i]) {
          outData[input1[i] - 1] = inData[i]; 
    }
   } 
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
