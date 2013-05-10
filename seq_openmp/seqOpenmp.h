#ifndef _SEQ_OPENMP_
#define _SEQ_OPENMP_

void seq_openmp();
void sequential_select(int N, int inData[], int outData[]);
void openmp_select(int N, int inData[], int outData[]);
bool validate(int N, int* openmp, int* target);
float toBW(int bytes, float sec);

#endif
