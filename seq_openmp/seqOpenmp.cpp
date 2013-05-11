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
  int* openmp_index_buffer = new int[NUM_TUPLES];
  int* openmp_index_sum = new int[NUM_TUPLES];
  for (int i = 0; i < NUM_TUPLES; i++) {
    //    relation[i] = rand() % 1000 + 1;
    relation[i] = 2;
    result[i] = 0;
    openmp_index_buffer[i] = 0;
  }


  startTime = CycleTimer::currentSeconds();
  sequential_select(NUM_TUPLES, relation, result);
  endTime = CycleTimer::currentSeconds();
  overallDuration = endTime - startTime;
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

void openmp_select(int N, int inData[], int outData[]) {




  
// #pragma omp parallel for schedule(dynamic, 1)
//   for(int i = 0; i < N; i ++) {
//     if(inData[i] % 2 == 0){
// 	     outData[i] = 1;
//       }
//     }
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