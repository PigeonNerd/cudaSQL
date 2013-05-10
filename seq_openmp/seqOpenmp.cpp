#include <stdlib.h>
#include <stdio.h>
#include <cstdlib>
#include <string>
#include <getopt.h>
#include "CycleTimer.h"
#include "seqOpenmp.h"

#define magnitude 20

void seq_openmp() {

  int base = 1;

  #pragma omp parallel for
  for (int i = 0; i < magnitude; i++) {
    base <<= 1;
  }
  printf("%d\n", base);

  int NUM_TUPLES = 12 * base;
  int* relation = new int[NUM_TUPLES];
  int* result = new int[NUM_TUPLES];
  int* openmp_result = new int[NUM_TUPLES];

  #pragma omp parallel for
  for (int i = 0; i < NUM_TUPLES; i++) {
    relation[i] = rand() % 1000 + 1;
    result[i] = 0;
  }

  sequential_select(NUM_TUPLES, relation, result);
  double startTime = CycleTimer::currentSeconds();
  openmp_select(NUM_TUPLES, relation, openmp_result);
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

void openmp_select(int N, int inData[], int outData[]) {
  int counter = 0;
  #pragma omp parallel for
  for(int i = 0; i < N; i ++) {
    if(inData[i] % 2 == 0) {
      outData[counter] = inData[i];
      #pragma omp atomic
      counter++;
    }
  }
}

bool validate(int N, int* openmp, int* target) {
   for(int i = 0 ; i < N; i ++) {
    if(openmp[i] != target[i]) {
      printf("ERROR: Result tuple %d not match, Expected: %d, Actual: %d\n",i, openmp[i] \
	     ,
	     target[i]);
      return false;
    }
  }
  printf("SELECT PASS\n");
  return true;
}

float toBW(int bytes, float sec) {
  return static_cast<float>(bytes) / (1024. * 1024. * 1024.) / sec;
}
