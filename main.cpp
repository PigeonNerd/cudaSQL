#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>
#include <cstdlib>
#include "CycleTimer.h"
#define magnitude 20 

void printCudaInfo();
void primitive_select(int N, int inData[], int outData[]);
void primitive_scan(int N, int inData[], int outData[]);
void sequential_select(int N , int inData[], int outData[]);
bool validate(int N, int* sequential, int* target); 
float toBW(int bytes, float sec);

int main(int argc, char** argv) {
    
    int base = 1;
    for(int i = 0; i < magnitude; i ++) {
        base <<= 1; 
    }
    printf("%d\n", base);

	int NUM_TUPPLES  = 20 * 1000 * 1000; 
	int* relation = new int[NUM_TUPPLES];
	int* cuda_result = new int[NUM_TUPPLES];
	int* sequential_result = new int[NUM_TUPPLES];
     for(int i = 0; i < NUM_TUPPLES; i++) {
		relation[i] = rand() % 100000 + 1;
		cuda_result[i] = 0;
	}
	primitive_select(NUM_TUPPLES, relation, cuda_result);
    double startTime = CycleTimer::currentSeconds();
    sequential_select(NUM_TUPPLES, relation, sequential_result); 
    double endTime = CycleTimer::currentSeconds();
    double overallDuration = endTime - startTime;
    printf("Sequential overall: %.3f ms\t\t[%.3f GB/s]\n", 1000.f * overallDuration, toBW( NUM_TUPPLES * sizeof(int) * 2, overallDuration));
	validate(NUM_TUPPLES, sequential_result, cuda_result);
   //primitive_scan(0, NULL, NULL); 
     
     return 0;
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

bool validate(int N, int* sequential, int* target) {
    for(int i = 0 ; i < N; i ++) {
        if(sequential[i] != target[i]) {
            printf("ERROR: Result tuple not match, Expected: %d, Actual: %d\n", sequential[i], 
                    target[i]);
            return false;
        }
    }
    printf("PASS\n");
    return true;
}

float toBW(int bytes, float sec) {
      return static_cast<float>(bytes) / (1024. * 1024. * 1024.) / sec;
}
