#include <stdio.h>
#include <algorithm>
#include <getopt.h>

#include "CycleTimer.h"
#include "seq_ispc.h"


void sequential_select(int N, int inData[], int outData[]);

using namespace ispc;

int main() {
    
    printf("Hello world\n");

	return 0;
}





void sequential_select_dumb(int N, int inData[], int outData[]) {
    int counter = 0;
    for(int i = 0; i < N; i ++) {
        if(inData[i] % 2 == 0) {
            outData[counter] = inData[i]; 
            counter++;
        }
    }
}
