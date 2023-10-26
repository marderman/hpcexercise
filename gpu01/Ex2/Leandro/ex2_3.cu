// 1gbyte = 1073741824 bytes, 1 int = 4 byte, 1gb = 268435456 int
// 1mbyte = 1048576 bytes, 1 int = 4 byte, 1mb = 262144 int
// 1kbyte = 1024 bytes, 1 int = 4 byte, 1kb = 256 int
#include <stdio.h>
#include <math.h>
#include "chTimer.h"

chTimerTimestamp start,stop;
double seconds;
void *dmem;
void *hmem;
const int count = 10;
double times[count];

double calculate_avg_time(double times[], int n){
    double temp = 0;
    for(int i = 0; i< n; i++){
        temp+=times[i];
    }
    return temp = temp/n;
}

void gpu_test(int amount, int size, bool pinned, bool todevice){
if(pinned == false && todevice == false){
    printf("Read from Device (non pinned memory)\n");
    for (size_t i = 0; i < count; i++)
    {
        hmem = malloc(amount * 256 * pow(1024,size)* sizeof(int));
        cudaMalloc(&dmem, amount * 256 * pow(1024,size)* sizeof(int));
                chTimerGetTime( &start );
                cudaMemcpy(hmem, dmem, amount * 256 * pow(1024,size)* sizeof(int), cudaMemcpyDeviceToHost);
                cudaDeviceSynchronize();
                chTimerGetTime( &stop );
                seconds = 1e12*chTimerElapsedTime( &start, &stop );
                times[i] = seconds/1e6;
                //printf( "%.2f us\n", (seconds/1e6));
        cudaFree(dmem);
        free(hmem);
    }
    printf( "Average %.2f us\n", calculate_avg_time(times, sizeof(times)/sizeof(times[0])));
}
if(pinned == false && todevice == true){
    printf("Write to Device (non pinned memory)\n");
    for (size_t i = 0; i < count; i++)
    {
        hmem = malloc(amount * 256 * pow(1024,size)* sizeof(int));
        cudaMalloc(&dmem, amount * 256 * pow(1024,size)* sizeof(int));
                chTimerGetTime( &start );
                cudaMemcpy(dmem, hmem, amount * 256 * pow(1024,size)* sizeof(int), cudaMemcpyHostToDevice);
                cudaDeviceSynchronize();
                chTimerGetTime( &stop );
                seconds = 1e12*chTimerElapsedTime( &start, &stop );
                times[i] = seconds/1e6;
                //printf( "%.2f us\n", (seconds/1e6));
        cudaFree(dmem);
        free(hmem);
    }
    printf( "Average %.2f us\n", calculate_avg_time(times, sizeof(times)/sizeof(times[0])));
}
if(pinned == true && todevice == false){
    printf("Read from Device (pinned memory)\n");
    for (size_t i = 0; i < count; i++)
    {
        cudaMallocHost(&hmem, amount * 256 * pow(1024,size)* sizeof(int));
        cudaMalloc(&dmem, amount * 256 * pow(1024,size)* sizeof(int));
                chTimerGetTime( &start );
                cudaMemcpy(hmem, dmem, amount * 256 * pow(1024,size)* sizeof(int), cudaMemcpyDeviceToHost);
                cudaDeviceSynchronize();
                chTimerGetTime( &stop );
                seconds = 1e12*chTimerElapsedTime( &start, &stop );
                times[i] = seconds/1e6;
                //printf( "%.2f us\n", (seconds/1e6));
        cudaFree(dmem);
        cudaFree(hmem);   
    }
    printf( "Average %.2f us\n", calculate_avg_time(times, sizeof(times)/sizeof(times[0])));
}
if(pinned == true && todevice == true){
    printf("Write to Device (pinned memory)\n");
    for (size_t i = 0; i < count; i++)
    {
        cudaMallocHost(&hmem, amount * 256 * pow(1024,size)* sizeof(int));
        cudaMalloc(&dmem, amount * 256 * pow(1024,size)* sizeof(int));
                chTimerGetTime( &start );
                cudaMemcpy(dmem, hmem, amount * 256 * pow(1024,size)* sizeof(int), cudaMemcpyHostToDevice);
                cudaDeviceSynchronize();
                chTimerGetTime( &stop );
                seconds = 1e12*chTimerElapsedTime( &start, &stop );
                times[i] = seconds/1e6;
                //printf( "%.2f us\n", (seconds/1e6));
        cudaFree(dmem);
        cudaFree(hmem);  
    }
    printf( "Average %.2f us\n", calculate_avg_time(times, sizeof(times)/sizeof(times[0])));
}
}

int main()
{
    bool pinned, todevice;
    for (size_t i = 0; i < 4; i++)
    {
        switch(i){
        case 0:
        pinned = false;
        todevice = false;
        break;
        case 1:
        pinned = false;
        todevice = true;
        break;
        case 2:
        pinned = true;
        todevice = false; 
        break;
        case 3:
        pinned = true;
        todevice = true;
        break;    
        }
        gpu_test(1,0,pinned, todevice);  //1kb
        gpu_test(128,0,pinned, todevice); //128kb
        gpu_test(1,1,pinned, todevice);  //1mb
        gpu_test(10,1,pinned, todevice);  //10mb
        gpu_test(128,1,pinned, todevice);  //128mb
        gpu_test(256,1,pinned, todevice);  //256mb
        gpu_test(512,1,pinned, todevice);  //512mb
        gpu_test(640,1,pinned, todevice);  //640mb
        gpu_test(768,1,pinned, todevice);  //768mb
        gpu_test(1,2,pinned, todevice);  //1gb
    }
    
    return 0;
}