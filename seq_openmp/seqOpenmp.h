#ifndef _SEQ_OPENMP_
#define _SEQ_OPENMP_

void seq_openmp();
void sequential_select(int N, int inData[], int outData[]);
void openmp_select_tuples(int N, int bufferData[], int inData[], int outData[], int sumData[]);
void openmp_select(int N, int inData[], int outData[]);
void openmp_prefixsum(int N, int sumData[]);
bool validate(int N, int* openmp, int* target);
float toBW(int bytes, float sec);

#endif
