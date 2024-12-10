#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//�����ڴ�ͱ����ڴ棨surface memory��ʵ������ȫ���ڴ��һ��������̬��ȫ���ڴ汻��Ϊ�����ڴ棨�����ڴ棩��
//����Ķ���д��������ͨ��ר�ŵ�texture cache�������棩���У���ʵ��Ϊ�������������

//Texture Memory: 
//...Optimized for read-only access to memory in 2D or 3D spatial locality patterns.
//...Supports built-in hardware-accelerated features like linear interpolation and normalized coordinate addressing.

//Surface Memory: 
//...Optimized for read-write operations to global memory.
//...does not support interpolation or normalized addressing like texture memory

//struct cudaTextureDesc
//{
//    enum cudaTextureAddressMode addressMode[3];
//    enum cudaTextureFilterMode  filterMode;
//    enum cudaTextureReadMode    readMode;
//    int                         sRGB;
//    int                         normalizedCoords;
//    unsigned int                maxAnisotropy;
//    enum cudaTextureFilterMode  mipmapFilterMode;
//    float                       mipmapLevelBias;
//    float                       minMipmapLevelClamp;
//    float                       maxMipmapLevelClamp;
//};

class TextureSurfaceMemory {
public:
    int TestTextureMemory();
    int TestSurfaceMemory();
};