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

#ifndef MIN
#define MIN(a,b) ((a < b) ? a : b)
#endif
#ifndef MAX
#define MAX(a,b) ((a > b) ? a : b)
#endif

#define DATA_SIZE (1<<20)//1048576
int THREAD_NUM  = (1<<8);//256
int BLOCK_NUM   = (1<<5);//32
int size_warp;//warp线程束的尺寸，即一个束包含多少个线程
int size_mp;//多处理器MultiProcessor的个数
int size_thread_max_per_block;//单个块最大线程数
int size_block_max_per_dimention;//单个维度最大块数

float time_cost_min=1000000.0f;
int size_thread_best;
int size_block_best;

int data[DATA_SIZE];

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
// Beginning of GPU Architecture definitions
inline int _ConvertSMVer2Cores(int major, int minor)
{
    // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
    typedef struct {
       int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
       int Cores;
    } sSMtoCores;

	sSMtoCores nGpuArchCoresPerSM[] = 
	{ { 0x10,  8 }, // Tesla Generation (SM 1.0) G80 class
	  { 0x11,  8 }, // Tesla Generation (SM 1.1) G8x class
	  { 0x12,  8 }, // Tesla Generation (SM 1.2) G9x class
	  { 0x13,  8 }, // Tesla Generation (SM 1.3) GT200 class
	  { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
	  { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
	  { 0x30, 192}, // Fermi Generation (SM 3.0) GK10x class
	  {   -1, -1 }
	};

    int index = 0;
    while (nGpuArchCoresPerSM[index].SM != -1) {
       if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor) ) {
          return nGpuArchCoresPerSM[index].Cores;
       }	
       index++;
    }
    printf("MapSMtoCores undefined SM %d.%d is undefined (please update to the latest SDK)!\n", major, minor);
    return -1;
}
// 硬件拥有最大的浮点计算能力GFLOPS
int gpuGetMaxGflopsDeviceId(float& fGFLOPS)
{
    int current_device     = 0, sm_per_multiproc  = 0;
    int max_compute_perf   = 0, max_perf_device   = 0;
    int device_count       = 0, best_SM_arch      = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceCount( &device_count );
    
    // Find the best major SM Architecture GPU device
    while (current_device < device_count)
    {
        cudaGetDeviceProperties( &deviceProp, current_device );
        if (deviceProp.major > 0 && deviceProp.major < 9999)
        {
            best_SM_arch = MAX(best_SM_arch, deviceProp.major);
        }
        current_device++;
    }

    // Find the best CUDA capable GPU device
    current_device = 0;
	while( current_device < device_count )
	{
		cudaGetDeviceProperties( &deviceProp, current_device );
		if (deviceProp.major == 9999 && deviceProp.minor == 9999)
		{
			sm_per_multiproc = 1;
		}
		else
		{
			sm_per_multiproc = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
		}

		int compute_perf  = deviceProp.multiProcessorCount * sm_per_multiproc * deviceProp.clockRate;
		// clockRate指shader的频率，单位是kHz，即"Clock frequency in kilohertz "，参考：http://developer.download.nvidia.com/compute/cuda/4_2/rel/toolkit/docs/online/structcudaDeviceProp_dee14230e417cb3059d697d6804da414.html#dee14230e417cb3059d697d6804da414

		if( compute_perf  > max_compute_perf )
		{
			// If we find GPU with SM major > 2, search only these
			if ( best_SM_arch > 2 )
			{
				// If our device==dest_SM_arch, choose this, or else pass
				if (deviceProp.major == best_SM_arch)
				{
					max_compute_perf  = compute_perf;
					max_perf_device   = current_device;
				}
			}
			else
			{
				max_compute_perf  = compute_perf;
				max_perf_device   = current_device;
				size_warp = deviceProp.warpSize;
				size_mp = deviceProp.multiProcessorCount;
				size_thread_max_per_block = deviceProp.maxThreadsPerBlock;
				size_block_max_per_dimention = deviceProp.maxGridSize[0];
				printf("cuda设备关键属性如下：\n");
				printf("warp尺寸：%d, shader频率: %d \n", size_warp, deviceProp.clockRate);
				printf("mp个数：%d, 每个mp所含sp个数：%d, sp核数: %d \n", size_mp, sm_per_multiproc, size_mp * sm_per_multiproc);
				printf("单个块最大线程数：%d \n", size_thread_max_per_block);
				printf("单个维度最大块数：%d \n", size_block_max_per_dimention);
			}
		}
		++current_device;
	}
	fGFLOPS = max_compute_perf * 1.0e-6;
	return max_perf_device;
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
#if 1
	float fGFLOPS = 0.0f;
	i = gpuGetMaxGflopsDeviceId( fGFLOPS );
    printf("计算能力估算公式=sp核数 * shader频率 \n\
			计算能力粗略估算: %0.2f GFLOPS\n", fGFLOPS);
#else
    for(i = 0; i < count; i++) {
        cudaDeviceProp prop;
        if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            if(prop.major >= 1) {
                break;
            }
        }
    }
#endif
    if(i == count) {
        fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
        return false;
    }

    cudaSetDevice(i);

    return true;
}



void GenerateNumbers(int *number, int size)
{
    for(int i = 0; i < size; i++) {
        number[i] = rand() % 10;
    }
}


__global__ static void sumOfSquares(int *num, int* result, clock_t* time, int tread_num, int block_num)
{
	const int tid = threadIdx.x;
	const int bid = blockIdx.x;

	int sum = 0;
	int i;

	if(tid == 0) time[bid] = clock();
	for(i = tid + bid * tread_num; i < DATA_SIZE; i+= tread_num * block_num) {
		sum += num[i] * num[i];
	}

	result[tid + bid * tread_num] = sum;
	if(tid == 0) time[bid + block_num] = clock();
}

void runCUDA()
{

	int* gpudata, *result;
	clock_t* time;
	cudaMalloc((void**) &gpudata, sizeof(int) * DATA_SIZE);
	cudaMalloc((void**) &result, sizeof(int) * THREAD_NUM * BLOCK_NUM);
	cudaMalloc((void**) &time, sizeof(clock_t) * BLOCK_NUM * 2);
	cudaMemcpy(gpudata, data, sizeof(int) * DATA_SIZE, cudaMemcpyHostToDevice);

	sumOfSquares<<<BLOCK_NUM, THREAD_NUM, 0>>>(gpudata, result, time, THREAD_NUM, BLOCK_NUM);

	int *sum = new int[THREAD_NUM * BLOCK_NUM];
	clock_t *time_used = new clock_t[BLOCK_NUM * 2];
	cudaMemcpy(sum, result, sizeof(int) * THREAD_NUM * BLOCK_NUM , cudaMemcpyDeviceToHost);
	cudaMemcpy(time_used, time, sizeof(clock_t) * BLOCK_NUM * 2, cudaMemcpyDeviceToHost);
	cudaFree(gpudata);
	cudaFree(result);

	int final_sum = 0;
	for(int i = 0; i < BLOCK_NUM * THREAD_NUM; i++) {
		final_sum += sum[i] ;
	}
	printf("sum（GPU）: %d\n", final_sum);
	delete[] sum;

	clock_t min_start, max_end;
	min_start = time_used[0];
	max_end = time_used[BLOCK_NUM];
	for(int i = 1; i < BLOCK_NUM; i++) {
		if(min_start > time_used[i])
			min_start = time_used[i];
		if(max_end < time_used[i + BLOCK_NUM])
			max_end = time_used[i + BLOCK_NUM];
	}
	int time_final = max_end - min_start;
	float time_final_per_element = time_final*1.0f/DATA_SIZE;
	printf("time: %d, time/n: %.4f\n", time_final, time_final_per_element);
	delete[] time_used;

	if( time_final_per_element < time_cost_min )
	{
		time_cost_min = time_final_per_element;
		
		if( size_thread_best!=THREAD_NUM )
			size_thread_best = THREAD_NUM;
		
		if( size_block_best!=BLOCK_NUM )
			size_block_best = BLOCK_NUM;
	}

	if ( THREAD_NUM * BLOCK_NUM == DATA_SIZE )
	{
		printf("每个线程计算一个元素。\n\n");
	}
}

int main(int argc, char **argv)
{
	//shrQAStart(argc, argv);
	 if(!InitCUDA()) {
        return 0;
    }

    printf("CUDA initialized.\n\n");

	 GenerateNumbers(data, DATA_SIZE);

	 printf(" 块数维持32，以32(warp的尺寸)为倍数变换线程数\n\n");
	 int size_thread_init = int( log2((float)size_warp) );// i从log2(32) = 5开始
	 int size_thread_end = int( log2((float)size_thread_max_per_block) );// i到log2(512) = 9结束
	 BLOCK_NUM = 1 << 5;
	 for (int i= size_thread_init;i<=size_thread_end;i++) 
	 {
		 THREAD_NUM = 1 << i;	
		 runCUDA();
		 printf(" THREAD_NUM = %d , BLOCK_NUM = %d\n\n", THREAD_NUM, BLOCK_NUM);
	 }

	 for (int i=size_thread_init;i<size_thread_end-1;i++)
	 {
		 THREAD_NUM = (1 << i)*3;	
		 runCUDA();
		 printf(" THREAD_NUM = %d , BLOCK_NUM = %d\n\n", THREAD_NUM, BLOCK_NUM);
	 }


	 printf(" 线程数维持256，以16(mp的个数)为倍数变换块数\n\n");
	 int size_block_init = int( log2((float)size_mp) );// i从log2(16) = 4开始
	 int size_block_end = int( log2((float)size_block_max_per_dimention) ); // i 到 log2(64k) = 16结束
	 THREAD_NUM = 1 << 8;
	 for (int i=size_block_init;i<size_block_end;i++)
	 {
		 BLOCK_NUM = 1 << i;	
		 runCUDA();
		 printf(" THREAD_NUM = %d , BLOCK_NUM = %d\n\n", THREAD_NUM, BLOCK_NUM);
	 }

	 for (int i=size_block_init;i<size_block_end-1;i++)
	 {
		 BLOCK_NUM = (1 << i)*3;	
		 runCUDA();
		 printf(" THREAD_NUM = %d , BLOCK_NUM = %d\n\n", THREAD_NUM, BLOCK_NUM);
	 }

	 printf(" 找出最优化的块数和线程数，进行运算。\n\n");
	 printf(" best size of thread = %d\n\n", size_thread_best);
	 printf(" best size of block = %d\n\n", size_block_best);
	 BLOCK_NUM = size_block_best;
	 THREAD_NUM = size_thread_best;
	 runCUDA();

	 // cpu计算
	 int final_sum = 0;
	 for(int i = 0; i < DATA_SIZE; i++) {
		 final_sum += data[i] * data[i];
	 }
	 printf("sum（CPU）: %d of %d squares\n", final_sum, DATA_SIZE);
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
	system("pause");
}
