#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//纹理内存和表面内存（surface memory）实质上是全局内存的一个特殊形态，全局内存被绑定为纹理内存（表面内存），
//对其的读（写）操作将通过专门的texture cache（纹理缓存）进行，其实称为纹理缓存更加贴切

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