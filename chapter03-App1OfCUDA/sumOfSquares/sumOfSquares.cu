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

//#include <shrQATest.h>
#include <cuda_runtime.h>

using namespace std;

bool g_bQATest = false;

#ifdef _WIN32
   #define STRCASECMP  _stricmp
   #define STRNCASECMP _strnicmp
#else
   #define STRCASECMP  strcasecmp
   #define STRNCASECMP strncasecmp
#endif

#define ASSERT(x, msg, retcode) \
    if (!(x)) \
    { \
        cout << msg << " " << __FILE__ << ":" << __LINE__ << endl; \
        return retcode; \
    }

__global__ void sequence_gpu(int *d_ptr, int length)
{
    int elemID = blockIdx.x * blockDim.x + threadIdx.x;
    if (elemID < length)
    {
        d_ptr[elemID] = elemID;
    }
}


void sequence_cpu(int *h_ptr, int length)
{
    for (int elemID=0; elemID<length; elemID++)
    {
        h_ptr[elemID] = elemID;
    }
}

void processArgs(int argc, char **argv)
{
    for (int i=1; i < argc; i++) {
        if((!STRNCASECMP((argv[i]+1), "noprompt", 8)) || (!STRNCASECMP((argv[i]+2), "noprompt", 8)) )
        {
            g_bQATest = true;
        }
    }
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
    for(i = 0; i < count; i++) {
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

    return true;
}

#define DATA_SIZE 1048576

int data[DATA_SIZE];

void GenerateNumbers(int *number, int size)
{
    for(int i = 0; i < size; i++) {
        number[i] = rand() % 10;
    }
}

__global__ static void sumOfSquares(int *num, int* result)
{
    int sum = 0;
    int i;
    for(i = 0; i < DATA_SIZE; i++) {
        sum += num[i] * num[i];
    }

    *result = sum;
}

int main(int argc, char **argv)
{
	//shrQAStart(argc, argv);
	 if(!InitCUDA()) {
        return 0;
    }

    printf("CUDA initialized.\n");

	 GenerateNumbers(data, DATA_SIZE);

    int* gpudata, *result;
    cudaMalloc((void**) &gpudata, sizeof(int) * DATA_SIZE);
    cudaMalloc((void**) &result, sizeof(int));
    cudaMemcpy(gpudata, data, sizeof(int) * DATA_SIZE, cudaMemcpyHostToDevice);

	sumOfSquares<<<1, 1, 0>>>(gpudata, result);

    int sum;
    cudaMemcpy(&sum, result, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(gpudata);
    cudaFree(result);

    printf("sum: %d\n", sum);

	sum = 0;
    for(int i = 0; i < DATA_SIZE; i++) {
        sum += data[i] * data[i];
    }
    printf("sum (CPU): %d\n", sum);

#if 0
    cout << "CUDA Runtime API template" << endl;
    cout << "=========================" << endl;
    cout << "Self-test started" << endl;

    const int N = 100;

    processArgs(argc, argv);

    int *d_ptr;
    ASSERT(cudaSuccess == cudaMalloc    (&d_ptr, N * sizeof(int)), "Device allocation of " << N << " ints failed", -1);

    int *h_ptr;
    ASSERT(cudaSuccess == cudaMallocHost(&h_ptr, N * sizeof(int)), "Host allocation of "   << N << " ints failed", -1);

    cout << "Memory allocated successfully" << endl;

    dim3 cudaBlockSize(32,1,1);
    dim3 cudaGridSize((N + cudaBlockSize.x - 1) / cudaBlockSize.x, 1, 1);
    sequence_gpu<<<cudaGridSize, cudaBlockSize>>>(d_ptr, N);
    ASSERT(cudaSuccess == cudaGetLastError(), "Kernel launch failed", -1);
    ASSERT(cudaSuccess == cudaDeviceSynchronize(), "Kernel synchronization failed", -1);

    sequence_cpu(h_ptr, N);

    cout << "CUDA and CPU algorithm implementations finished" << endl;

    int *h_d_ptr;
    ASSERT(cudaSuccess == cudaMallocHost(&h_d_ptr, N * sizeof(int)), "Host allocation of " << N << " ints failed", -1);
    ASSERT(cudaSuccess == cudaMemcpy(h_d_ptr, d_ptr, N * sizeof(int), cudaMemcpyDeviceToHost), "Copy of " << N << " ints from device to host failed", -1);
    bool bValid = true;
    for (int i=0; i<N && bValid; i++)
    {
        if (h_ptr[i] != h_d_ptr[i])
        {
            bValid = false;
        }
    }

    ASSERT(cudaSuccess == cudaFree(d_ptr),       "Device deallocation failed", -1);
    ASSERT(cudaSuccess == cudaFreeHost(h_ptr),   "Host deallocation failed",   -1);
    ASSERT(cudaSuccess == cudaFreeHost(h_d_ptr), "Host deallocation failed",   -1);

    cout << "Memory deallocated successfully" << endl;
    cout << "TEST Results " << endl;
#endif  
    //shrQAFinishExit(argc, (const char **)argv, (bValid ? QA_PASSED : QA_FAILED));
}
