#include "../common/book.h"
#include "assert.h"
#define N 65000

__global__ void add(int *a, int *b, int *c){
  int tid = blockIdx.x;
  if(tid < N){
    c[tid] = a[tid] + b[tid];
  }
}

int main(void){
  int a[N], b[N], c[N];
  int *dev_a, *dev_b, *dev_c;
  // allocate device memory
  cudaMalloc((void **)&dev_a, N*sizeof(int));
  cudaMalloc((void **)&dev_b, N*sizeof(int));
  cudaMalloc((void **)&dev_c, N*sizeof(int));
  
  // init array
  for(int i=0; i < N; i++){
    a[i] = -i;
    b[i] = i*i;
  }

  cudaMemcpy(dev_a, a, N*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b, N*sizeof(int), cudaMemcpyHostToDevice);


  add<<<N,1>>>(dev_a, dev_b, dev_c);

  cudaMemcpy(c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost);

  // check the results
  for(int i=0; i < N; i++){
    assert(c[i] == a[i] + b[i]);
  }

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);
  return 0;
}
