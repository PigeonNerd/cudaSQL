#include <fcntl.h>
#include <getopt.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <unistd.h>
#include "CycleTimer.h"
#include "seqOpenmp.h"
extern void seq_openmp();
int main() {
  seq_openmp();
  return 0;
}
