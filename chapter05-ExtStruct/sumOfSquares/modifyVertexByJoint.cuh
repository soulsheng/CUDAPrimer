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

 /* This example demonstrates how to use the Cuda OpenGL bindings with the
  * runtime API.
  * Device code.
  */

#ifndef _modifyVertexByJoint_KERNEL_H_
#define _modifyVertexByJoint_KERNEL_H_

#include "math.h"
#include "StructMS3D.h"

// 必须要有.字节对齐选项.
#pragma pack(push, packing)
#pragma pack(4)


//-------------------------------------------------------------
//- DMs3dKeyFrame
//- Rotation/Translation information for joints
struct DMs3dKeyFrame
{
	float m_fTime;     //Time at which keyframe is started
	float m_fParam[3]; //Translation or Rotation values
	void clone(MS3DKeyframe& sref)
	{
		m_fTime = sref.m_fTime;
		memcpy( &m_fParam,&sref.m_fParam, sizeof(float3) );
	}
};

//-------------------------------------------------------------
//- DMs3dJoint
//- Bone Joints for animation
struct DMs3dJoint
{
public:
	//Data from file
	unsigned char m_ucpFlags;             //Editor flags
	char1 m_cName[32];                     //Bone name
	char1 m_cParent[32];                   //Parent name
	float3 m_fRotation;                 //Starting rotation
	float3 m_fPosition;                 //Starting position
	unsigned short m_usNumRotFrames;      //Numbee of rotation frames
	unsigned short m_usNumTransFrames;    //Number of translation frames

	DMs3dKeyFrame * m_RotKeyFrames;       //Rotation keyframes
	DMs3dKeyFrame * m_TransKeyFrames;     //Translation keyframes

	//Data not loaded from file
	short m_sParent;                     //Parent joint index

	float m_matLocal[16];
	float m_matAbs[16];
	float m_matFinal[16];

	unsigned short m_usCurRotFrame;
	unsigned short m_usCurTransFrame;
#if 0
	unsigned int	_idVBORotFrame;
	unsigned int	_idVBOTransFrame;

	cudaGraphicsResource *_resCudaVBORotFrame;
	cudaGraphicsResource *_resCudaVBOTransFrame;
#endif
	//Clean up after itself like usual
	DMs3dJoint()
	{
		m_RotKeyFrames = 0;
		m_TransKeyFrames = 0;
		m_usCurRotFrame = 0;
		m_usCurTransFrame = 0;
	}
	~DMs3dJoint()
	{
		if(m_RotKeyFrames)
		{
			delete [] m_RotKeyFrames;
			m_RotKeyFrames = 0;
		}
		if(m_TransKeyFrames)
		{
			delete [] m_TransKeyFrames;
			m_TransKeyFrames = 0;
		}
	}

};


#pragma pack(pop, packing)


#endif // #_modifyVertexByJoint_KERNEL_H_
