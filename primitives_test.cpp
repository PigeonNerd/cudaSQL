#include <stdio.h>
#include <stdlib.h>
#include <map>
#include <vector>
#include <string>

#define PARALLEL_THREADS_PER_CPU 4
#define PARALLEL_MAX_SERVERS 2
#define uint unsigned int

typedef struct query_blocks {
  int blockId;
  //store query block in some form
  int start; //start query at index
  int end; //end query at index
} queryBlocks;

static struct Primitive_Select {

  int num_workers;
  int num_pending_queries;

} select;

void primitive_select(const query_block& params) {
  select.num_workers = 0;

  pthread_t thread1;
  pthread_t thread2;
  pthread_t thread3;

  pthread.create(&thread1, NULL, run_select, NULL);
  pthread.create(&thread2, NULL, run_select, NULL);
  pthread.create(&thread3, NULL, run_select, NULL);
}

void run_select(void* arg) {
  query_block* query = (query_block*) arg;
  query_block response((*query).get_tag());

  runSelect(*query, response);
}
