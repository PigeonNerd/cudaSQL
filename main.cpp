#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>
#include <cstdlib>
#include "CycleTimer.h"
#define magnitude 10 

void printCudaInfo();
void primitive_select(int N, int inData[], int outData[]);
void primitive_join(int N , int M);
void primitive_scan(int N, int inData[], int outData[]);
void sequential_select(int N , int inData[], int outData[]);
void streamTest();
void sequentialTest();
void test_select(); 
void test_join();
bool validate(int N, int* sequential, int* target); 
float toBW(int bytes, float sec);

int main(int argc, char** argv) {
    test_select(); 
    test_join(); 
    return 0;
}


void test_join() {
	int NUM_TUPPLES_A  =  20; 
	int NUM_TUPPLES_B  =  20; 
    int min = 1;
    int max = 20;
    // these initializatoin might depcrated since 
    // it is not convinient to access 2D array in cuda
    int** rel_a = new int*[NUM_TUPPLES_A];    
    int** rel_b = new int*[NUM_TUPPLES_B];
    int** out = new int*[NUM_TUPPLES_A * NUM_TUPPLES_B];

    for(int i = 0 ; i < NUM_TUPPLES_A; i ++) {
        rel_a[i] = new int[2];
    }
    for(int i = 0 ; i < NUM_TUPPLES_B; i ++) {
        rel_b[i] = new int[2];
    }
    for(int i = 0 ; i < NUM_TUPPLES_B; i ++) {
        out[i] = new int[3];
    }

    for(int i = 0 ; i < NUM_TUPPLES_A; i ++) {
        rel_a[i][0] = min + (rand() % (int)(max - min + 1));
        rel_a[i][1] = min + (rand() % (int)(max - min + 1));
        printf("a [%d , %d]\n", rel_a[i][0], rel_a[i][1]);
    }
    for(int i = 0 ; i < NUM_TUPPLES_B; i ++) {
        rel_b[i][0] = min + (rand() % (int)(max - min + 1));
        rel_b[i][1] = min + (rand() % (int)(max - min + 1));
        printf("b [%d , %d]\n", rel_b[i][0], rel_b[i][1]);
    }
    primitive_join(NUM_TUPPLES_A, NUM_TUPPLES_B);
}

void test_select() {
    int base = 1;
    for(int i = 0; i < magnitude; i ++) {
        base <<= 1; 
    }
    printf("%d\n", base);

	int NUM_TUPPLES  =  20 * 1000 * 1000; 
	int* relation = new int[NUM_TUPPLES];
	int* cuda_result = new int[NUM_TUPPLES];
	int* sequential_result = new int[NUM_TUPPLES];
     for(int i = 0; i < NUM_TUPPLES; i++) {
		relation[i] = rand() % 100000 + 1;
		cuda_result[i] = 0;
	}
	primitive_select(NUM_TUPPLES, relation, cuda_result);
    double startTime = CycleTimer::currentSeconds();
    for(int i = 0 ; i < 10 ; i++) {
        sequential_select(NUM_TUPPLES, relation, sequential_result); 
    }
    double endTime = CycleTimer::currentSeconds();
    double overallDuration = endTime - startTime;
    printf("Sequential overall: %.3f ms\t\t[%.3f GB/s]\n", 1000.f * overallDuration, toBW( NUM_TUPPLES * sizeof(int) * 2, overallDuration));
	validate(NUM_TUPPLES, sequential_result, cuda_result);
    // primitive_scan(0, NULL, NULL); 
    // streamTest(); 
    // sequentialTest();
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
