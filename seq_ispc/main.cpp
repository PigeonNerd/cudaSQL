#include <stdio.h>
#include <algorithm>
#include <getopt.h>

#include "CycleTimer.h"
#include "seq_ispc.h"
#define magnitude  25

void sequential_select_dumb(int N, int inData[], int outData[]);
float toBW(int bytes, float sec);
using namespace ispc;

int main() {
    double startTime;
    double endTime;
    double overallDuration;
    int base = 1;
         for(int i = 0; i < magnitude; i ++) {
              base <<= 1;
          }
          printf("%d\n", base);
          int NUM_TUPPLES  = 3 * base;
          int* relation = new int[NUM_TUPPLES];
          int* sequential_result = new int[NUM_TUPPLES];
          for(int i = 0; i < NUM_TUPPLES; i++) {
                 relation[i] = rand() % 1000 + 1;
           }
         
    startTime = CycleTimer::currentSeconds();      
    select_ispc_withtasks(NUM_TUPPLES, relation,  sequential_result);
    endTime = CycleTimer::currentSeconds();
    overallDuration = endTime - startTime;
     printf("ISPC Sequential overall: %.3f ms\t\t[%.3f GB/s]\n", 1000.f * overallDuration, toBW( NUM_TUPPLES * sizeof(int) * 2, overallDuration));

    startTime = CycleTimer::currentSeconds();      
	sequential_select_dumb(NUM_TUPPLES, relation, sequential_result);
    endTime = CycleTimer::currentSeconds();
    overallDuration = endTime - startTime;
     printf("DUMB Sequential overall: %.3f ms\t\t[%.3f GB/s]\n", 1000.f * overallDuration, toBW( NUM_TUPPLES * sizeof(int) * 2, overallDuration));
        
     return 0;
}

void sequential_select_dumb(int N, int inData[], int outData[]) {
    int counter = 0;
    for(int i = 0; i < N; i ++) {
        if(inData[i] > 500 ) {
            outData[counter] = inData[i]; 
            counter++;
        }
    }
}
float toBW(int bytes, float sec) {
     return static_cast<float>(bytes) / (1024. * 1024. * 1024.) / sec;
}
