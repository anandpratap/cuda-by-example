#include "../common/book.h"

__global__ void add(int a, int b, int *c){
  *c = a + b;
}

int main(void){

  // add two numbers a and b and save the results in c
  int c;
  int *dev_c;
  HANDLE_ERROR(cudaMalloc((void **)&dev_c, sizeof(int)));
  add<<<1,1>>>(2, 7, dev_c);
  HANDLE_ERROR(cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost));
  printf("2 + 7 = %d\n", c);
  cudaFree(dev_c);

  // get device properties
  cudaDeviceProp prop;
  int count;
  HANDLE_ERROR(cudaGetDeviceCount(&count));
  for(int i=0; i<count;i++){
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, i));
    printf("Device name: %s\n", prop.name);
    printf("Total global memory: %lu bytes\n", prop.totalGlobalMem);
    printf("Shared memory per block: %lu bytes\n", prop.sharedMemPerBlock);
    printf("Multiprocessor count: %d\n", prop.multiProcessorCount);
    printf("Max thread per block: %d\n", prop.maxThreadsPerBlock);
    printf("Max thread dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0],
	   prop.maxThreadsDim[1],
	   prop.maxThreadsDim[2]);
    printf("Max grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0],
	   prop.maxGridSize[1],
	   prop.maxGridSize[2]);
    printf("compute capability %d.%d\n", prop.major, prop.minor);
  }
  return 0;
}
