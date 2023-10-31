#include<stdio.h>
#include<chTimer.h>

const int cIterations = 1000000;

chTimerTimestamp start, stop;

void test(size_t size, int* dintArray, int* hintArray, cudaMemcpyKind direction) {

  //Host 2 Device
  for (size_t i = 0; i < cIterations; i++)
  {
    if (direction == cudaMemcpyHostToDevice) {
    cudaMemcpy(dintArray, hintArray, size*sizeof(int),direction);
    cudaDeviceSynchronize();
    }
    else if(direction == cudaMemcpyDeviceToHost) {
    cudaMemcpy(hintArray,dintArray,size*sizeof(int),direction);
    cudaDeviceSynchrqonize();
    }
  }

  
}

int main()
{   
  //Allocate Memory on Host and Device
  const int allocatedMemorySize = 1*1024*1024*1024; //1GB
  int *hintArray, *dintArray;

  printf("Measuring transfeir from Host to Device with malloc allocated pageable memory");
  
  for (size_t i = 0; i < 10; i++)
  {
    hintArray = (int*)malloc(allocatedMemorySize*sizeof(int));
    cudaMalloc(&dintArray,allocatedMemorySize*sizeof(int));
    
    chTimerGetTime( &start );
    test(allocatedMemorySize,dintArray,hintArray,cudaMemcpyHostToDevice);
    chTimerGetTime(&stop);
    free(hintArray);
    cudaFree(dintArray);
    {
    double microseconds = 1e6*chTimerElapsedTime( &start, &stop );
    double usPerLaunch = microseconds / (float) cIterations;
    printf( "%.4f us\n", usPerLaunch );
  }

  }


  printf("Measuring transfer from Device to Host with malloc allocated pageable memory");

  for (size_t i = 0; i< 10; i++)
  {
    hintArray = (int*)malloc(allocatedMemorySize*sizeof(int));
    cudaMalloc(&dintArray,allocatedMemorySize*sizeof(int));
    chTimerGetTime( &start );
    test(allocatedMemorySize,dintArray,hintArray,cudaMemcpyDeviceToHost);
    chTimerGetTime(&stop);
    free(hintArray);
    cudaFree(dintArray);
    {
    double microseconds = 1e6*chTimerElapsedTime( &start, &stop );
    double usPerLaunch = microseconds / (float) cIterations;
    printf( "%.4f us\n", usPerLaunch );
  }

  }
//Test with cudaMalloc 

  printf("Measuring transfer from Host to Device with cudaMallocHost pinned memory");
  for (size_t i = 0; i < 10; i++)
  {
    cudaMallocHost(&hintArray,allocatedMemorySize);
    cudaMalloc(&dintArray,allocatedMemorySize*sizeof(int));
    chTimerGetTime( &start );
    test(allocatedMemorySize,dintArray,hintArray,cudaMemcpyHostToDevice);
    chTimerGetTime(&stop);
    cudaFree(hintArray);
    cudaFree(dintArray);
    {
    double microseconds = 1e6*chTimerElapsedTime( &start, &stop );
    double usPerLaunch = microseconds / (float) cIterations;
    printf( "%.4f us\n", usPerLaunch );
  }
  }
    printf("Measuring transfer from Device to Host with cudaMallocHost pinned memory");

  for (size_t i = 0; i < 10; i++)
  {
    cudaMallocHost(&hintArray,allocatedMemorySize);
    cudaMalloc(&dintArray,allocatedMemorySize*sizeof(int));
    test(allocatedMemorySize,dintArray,hintArray,cudaMemcpyDeviceToHost);
    chTimerGetTime(&stop);
    cudaFree(hintArray);
    cudaFree(dintArray);
    {
    double microseconds = 1e6*chTimerElapsedTime( &start, &stop );
    double usPerLaunch = microseconds / (float) cIterations;
    printf( "%.4f us\n", usPerLaunch );
  }
  }
//

return 0;
}

