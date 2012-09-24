/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* Template project which demonstrates the basics on how to setup a project 
* example application, doesn't use cutil library.
*/

#include <stdio.h>
#include <string.h>
#include <iostream>
using namespace std;

#include <cuda_runtime.h>

#include <cutil.h>
#include <cutil_inline_runtime.h>   

#define DATA_SIZE (1<<20)//1048576
#define THREAD_NUM  (1<<6)//64
#define BLOCK_NUM    (1<<7)//128

#define TIMES_REPERT	(1<<0)
int data[DATA_SIZE];

// 测时方法参考：http://soulshengbbs.sinaapp.com/thread-12-1-1.html 《cuda测量时间的方法汇总》二、cutGetTimerValue
unsigned int hTimer ;
void timeBegin()
{
	cutilDeviceSynchronize() ;
	cutStartTimer(hTimer) ;
	cutResetTimer(hTimer);
}
void timeEnd(string msg)
{
	cutilDeviceSynchronize() ;
	cutStopTimer(hTimer) ;

	double Passed_Time = cutGetTimerValue(hTimer);

	printf("time（%s）: %.3f ms\n", msg.c_str(), Passed_Time);
}

clock_t clockBegin,clockEnd;
void timeBeginCPU()
{
	clockBegin = clock();
}
void timeEndCPU(string msg)
{
	clockEnd = clock();

	double Passed_Time = clockEnd - clockBegin;

	printf("time（%s）: %.3f ms\n", msg.c_str(), Passed_Time);
}

bool InitCUDA()
{
    int count;

    cudaGetDeviceCount(&count);
    if(count == 0) {
        fprintf(stderr, "There is no device.\n");
        return false;
    }

	int i;
    for( i = 0; i < count; i++) {
        cudaDeviceProp prop;
        if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            if(prop.major >= 1) {
                break;
            }
        }
    }

    if(i == count) {
        fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
        return false;
    }

    cudaSetDevice(i);
	
    printf("CUDA initialized.\n\n");
    return true;
}



void GenerateNumbers(int *number, int size)
{
    for(int i = 0; i < size; i++) {
        number[i] = rand() % 10;
    }
}


__global__ static void sumOfSquares(int *num, int* result, clock_t* time)
{
	const int tid = threadIdx.x;
	const int bid = blockIdx.x;
	
	int sum ;
	int i;
	if(tid == 0) time[bid] = clock();
	for(i = tid + bid * THREAD_NUM; i < DATA_SIZE; i+= THREAD_NUM * BLOCK_NUM) {
		sum += num[i] * num[i] ;
	}

	result[tid + bid * THREAD_NUM] = sum;
    if(tid == 0) time[bid + BLOCK_NUM] = clock();
}

void runCUDA()
{
	int* gpudata, *result;
	clock_t* time;
	cudaMalloc((void**) &time, sizeof(clock_t) * BLOCK_NUM * 2);
	cudaMalloc((void**) &gpudata, sizeof(int) * DATA_SIZE);
	cudaMalloc((void**) &result, sizeof(int) * THREAD_NUM * BLOCK_NUM);
	cudaMemcpy(gpudata, data, sizeof(int) * DATA_SIZE, cudaMemcpyHostToDevice);

	sumOfSquares<<<BLOCK_NUM, THREAD_NUM, 0>>>(gpudata, result, time);

	int sum[THREAD_NUM * BLOCK_NUM];
    clock_t time_used[BLOCK_NUM * 2];
    cudaMemcpy(&time_used, time, sizeof(clock_t) * BLOCK_NUM * 2, cudaMemcpyDeviceToHost);
	cudaMemcpy( &sum, result, sizeof(int) * THREAD_NUM * BLOCK_NUM , cudaMemcpyDeviceToHost);
	cudaFree(gpudata);
	cudaFree(result);

	int final_sum = 0;
	for(int i = 0; i < BLOCK_NUM * THREAD_NUM; i++) {
		final_sum += sum[i] ;
	}

	clock_t min_start, max_end;
    min_start = time_used[0];
    max_end = time_used[BLOCK_NUM];
    for(int i = 1; i < BLOCK_NUM; i++) {
        if(min_start > time_used[i])
            min_start = time_used[i];
        if(max_end < time_used[i + BLOCK_NUM])
            max_end = time_used[i + BLOCK_NUM];
    }
    printf("time: %d, time/n: %.4f\n", max_end - min_start, (max_end - min_start)*1.0f/DATA_SIZE);
}

void runCPU()
{
	int final_sum = 0;
	 for(int i = 0; i < DATA_SIZE; i++) {
		 final_sum += data[i] * data[i];
	 }
	//printf("sum（CPU）: %d of %d squares\n", final_sum, DATA_SIZE);
}

int main(int argc, char **argv)
{
	// 初始化cuda
	if(!InitCUDA()) {
		return 0;
	}

	// 数据初始化
	GenerateNumbers(data, DATA_SIZE);

	cutCreateTimer(&hTimer);

	
	// cuda计算
	runCUDA();
	
	// cpu计算
	runCPU();


	cutDeleteTimer(hTimer);

	//system("pause");
}
