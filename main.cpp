#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>
#include <cstdlib>

void printCudaInfo();
void primitive_select(int N, int inData[], int outData[]);
void primitive_scan(int N, int inData[], int outData[]);

int main(int argc, char** argv) {
	 int NUM_TUPPLES  = 512;
	int* relation = new int[NUM_TUPPLES];
	 int* result = new int[NUM_TUPPLES];
	for(int i = 0; i < NUM_TUPPLES; i++) {
		relation[i] = rand() % 100000 + 1;
		result[i] = 0;
		printf("%d, ", relation[i]);
	}
	printf("\n");
	
	primitive_select(NUM_TUPPLES, relation, result);

	for(int i = 0; i < NUM_TUPPLES; i++) {
			printf("%d, ", result[i]);
	}
	printf("\n");

	return 0;
}