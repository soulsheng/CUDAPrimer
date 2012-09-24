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

#include "modifyVertexByJoint.cuh"
#include "StructMS3D.h"

#define DATA_SIZE (1<<16)//64k 个点
#define ATTRIB_SIZE		(1<<6)//64 个骨骼

#define THREAD_NUM  (1<<7)//128
#define BLOCK_NUM    ((1<<4)*6)//96

#define TIMES_REPERT	(1<<0)


Ms3dVertexArrayElement pVertexArray[DATA_SIZE];
Ms3dVertexArrayElement pVertexArrayBackup[DATA_SIZE];
DMs3dJoint	pJoints[ATTRIB_SIZE];

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

__device__
void deviceTransformVetex(float* pos, float* mat)
{
	float* m_fMat = mat;

	float x = pos[0] * m_fMat[0] + 
		pos[1] * m_fMat[4] +
		pos[2] * m_fMat[8] + 
		m_fMat[12] ;

	float y = pos[0] * m_fMat[1] + 
		pos[1] * m_fMat[5] + 
		pos[2] * m_fMat[9] + 
		m_fMat[13] ;

	float z = pos[0] * m_fMat[2] + 
		pos[1] * m_fMat[6] + 
		pos[2] * m_fMat[10]+
		m_fMat[14] ;

	pos[0] = x;
	pos[1] = y;
	pos[2] = z;

}
__global__ void modifyVertexByJointInGPUKernel( Ms3dVertexArrayElement* pVertexArray, Ms3dVertexArrayElement* pVertexArrayBackup, 
	DMs3dJoint * pJoints, int nTriangleIndices , clock_t* time )
{
	int loop1 = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

	if ( loop1 >= nTriangleIndices )
	{
		return;
	}

	const int tid = threadIdx.x;
	const int bid = blockIdx.x;
	if(tid == 0) time[bid] = clock();

	Ms3dVertexArrayElement* pVert = NULL;
	Ms3dVertexArrayElement* pVertBackup = NULL;
	int nIdBone = 0;

	// 遍历三角面的三个顶点 

	// 第1个点
	pVert = (Ms3dVertexArrayElement*)(pVertexArray + 9 * (loop1 * 3));
	pVertBackup = (Ms3dVertexArrayElement*)(pVertexArrayBackup + 9 * (loop1 * 3));

		pVert->m_vVert[0] = pVertBackup->m_vVert[0] ;
		pVert->m_vVert[1] = pVertBackup->m_vVert[1] ;
		pVert->m_vVert[2] = pVertBackup->m_vVert[2] ;

	nIdBone = (int)( pVertBackup->m_fBone + 0.5f );

	nIdBone == -1? 1: deviceTransformVetex( pVert->m_vVert, pJoints[ nIdBone].m_matFinal );

	// 第2个点
	pVert = (Ms3dVertexArrayElement*)(pVertexArray + 9 * (loop1 * 3 + 1));
	pVertBackup = (Ms3dVertexArrayElement*)(pVertexArrayBackup + 9 * (loop1 * 3 + 1));

	pVert->m_vVert[0] = pVertBackup->m_vVert[0] ;
	pVert->m_vVert[1] = pVertBackup->m_vVert[1] ;
	pVert->m_vVert[2] = pVertBackup->m_vVert[2] ;

	nIdBone = (int)( pVertBackup->m_fBone + 0.5f );

	nIdBone == -1? 1: deviceTransformVetex( pVert->m_vVert, pJoints[ nIdBone].m_matFinal );


	// 第3个点
	pVert = (Ms3dVertexArrayElement*)(pVertexArray + 9 * (loop1 * 3 + 2));
	pVertBackup = (Ms3dVertexArrayElement*)(pVertexArrayBackup + 9 * (loop1 * 3 + 2));

	pVert->m_vVert[0] = pVertBackup->m_vVert[0] ;
	pVert->m_vVert[1] = pVertBackup->m_vVert[1] ;
	pVert->m_vVert[2] = pVertBackup->m_vVert[2] ;

	nIdBone = (int)( pVertBackup->m_fBone + 0.5f );

	nIdBone == -1? 1: deviceTransformVetex( pVert->m_vVert, pJoints[ nIdBone].m_matFinal );
   
	if(tid == 0) time[bid + BLOCK_NUM] = clock();

}
#if 0
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
#endif
//Round a / b to nearest higher integer value
int iDivUp(int a, int b){
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

// compute grid and thread block size for a given number of elements
void computeGridSize(int n, int blockSize, int &numBlocks, int &numThreads)
{
    numThreads = min(blockSize, n);
    numBlocks = iDivUp(n, numThreads);
}

void runCUDA()
{
	void *dVert, *dVertBackup, *dJoint;
	clock_t* time;
	cudaMalloc((void**) &time, sizeof(clock_t) * BLOCK_NUM * 2);
	cudaMalloc((void**) &dVert, sizeof(Ms3dVertexArrayElement) * DATA_SIZE);
	cudaMalloc((void**) &dVertBackup, sizeof(Ms3dVertexArrayElement) * DATA_SIZE);
	cudaMalloc((void**) &dJoint, sizeof(DMs3dJoint) * ATTRIB_SIZE);
	cudaMemcpy(dVert, pVertexArray, sizeof(Ms3dVertexArrayElement) * DATA_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(dVertBackup, pVertexArrayBackup, sizeof(Ms3dVertexArrayElement) * DATA_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(dJoint, pJoints, sizeof(DMs3dJoint) * ATTRIB_SIZE, cudaMemcpyHostToDevice);

	//sumOfSquares<<<BLOCK_NUM, THREAD_NUM, 0>>>(gpudata, result, time);
	int numThreads, numBlocks;
    computeGridSize(DATA_SIZE, 256, numBlocks, numThreads);
	modifyVertexByJointInGPUKernel<<< numBlocks, numThreads >>>
		( pVertexArray, pVertexArrayBackup, pJoints, DATA_SIZE, time);

	
    clock_t time_used[BLOCK_NUM * 2];
    cudaMemcpy(&time_used, time, sizeof(clock_t) * BLOCK_NUM * 2, cudaMemcpyDeviceToHost);
	cudaFree(dVert);
	cudaFree(dVertBackup);
	cudaFree(dJoint);

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
#if 0
	int final_sum = 0;
	 for(int i = 0; i < DATA_SIZE; i++) {
		 final_sum += data[i] * data[i];
	 }
	//printf("sum（CPU）: %d of %d squares\n", final_sum, DATA_SIZE);
#endif
}

int main(int argc, char **argv)
{
	// 初始化cuda
	if(!InitCUDA()) {
		return 0;
	}

	// 数据初始化
	//GenerateNumbers(data, DATA_SIZE);

	cutCreateTimer(&hTimer);

	
	// cuda计算
	runCUDA();
	
	// cpu计算
	runCPU();


	cutDeleteTimer(hTimer);

	//system("pause");
}
