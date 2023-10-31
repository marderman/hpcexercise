#include<stdio.h>
#include<chTimer.h>



using namespace std;

__global__
void memcpyKernel(){

}



int main()
{   
  const int cIterations = 1000000;
  //Allocate Memory on Host and Device
  size_t allocatedMemorySize = 1*1024*1024*1024; //1GB
  int *hintArray, *dintArray;

  hintArray = (int*)malloc(allocatedMemorySize*sizeof(int));
  cudaMalloc(&dintArray,allocatedMemorySize*sizeof(int));

  printf("Measuring transfeir from Host to Device with malloc allocated pageable memory");
  chTimerTimestamp start, stop;

  memcpy(dintArray, hintArray, allocatedMemorySize*sizeof(int));

  cudaDeviceSynchronize();

  {
    double microseconds = 1e6*chTimerElapsedTime( &start, &stop );
    double usPerLaunch = microseconds / (float) cIterations;
    printf( "%.2f us\n", usPerLaunch );
  }
//D2H

//Test with cudaMalloc

//
free(hintArray);
cudaFree(dintArray);
}
