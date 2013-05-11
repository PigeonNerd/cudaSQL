#ifndef _SEQ_OPENMP_H_
#define _SEQ_OPENMP_H_

void seq_openmp();
bool validate(int N, int* openmp, int* target);
void sequential_select(int N, int inData[], int outData[]);
void openmp_select(int N, int inData[], int outData[]);
float toBW(int bytes, float sec);
void prefix_sum(int n, int x[], int t[]);
static void exclusive_scan(const int n, int *data);
#endif
