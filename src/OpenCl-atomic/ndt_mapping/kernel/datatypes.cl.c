/**
 * Author:  Leonardo Solis, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2018
 * Author:  Thilo Gabel, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2020
 * License: Apache 2.0 (see attachached File)
 */

#ifndef RESOUTION
#define RESOLUTION 1.0f
#endif

#ifndef INV_RESOLUTION
#define INV_RESOLUTION 1.0f/RESOLUTION
#endif

#ifndef RADIUS
#define RADIUS 1.0f
#endif

#ifndef RADIUS_FINAL
#define RADIUS_FINAL 1.001f
#endif

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define EPHOS_ATOMICS

typedef struct Mat33 {
  double data[3][3];
} Mat33;

typedef struct Vec3 {
	double data[3];
} Vec3;

typedef struct VoxelCovariance {
	double data[3][3];
} VoxelCovariance;

typedef struct VoxelMean {
	double data[3];
} VoxelMean;

typedef struct {
    float data[4];
} PointXYZI;

typedef struct {
    VoxelCovariance invCovariance;
    VoxelMean mean;
	int pointListBegin;
#ifdef EPHOS_VOXEL_POINT_STORAGE
	int pointStorageLevel;
	PointXYZI pointStorage[EPHOS_VOXEL_POINT_STORAGE];
#endif
} Voxel;

typedef struct {
	VoxelCovariance invCovariance;
	VoxelMean mean;
	int point;
} PointVoxel;

typedef struct {
	float x;
	float y;
	float z;
	int iNext;
} PointQueue;

typedef struct VoxelGridCorner {
	float data[3];
} VoxelGridCorner;

typedef struct PackedVoxelGridCorner{
	int data[4];
} PackedVoxelGridCorner;

typedef struct VoxelGridDimension {
	int data[2];
} VoxelGridDimension;

typedef struct VoxelGridInfo {
	int cloudSize;
	int gridSize;
	VoxelGridCorner minCorner;
	VoxelGridCorner maxCorner;
	VoxelGridDimension gridDimension;
} VoxelGridInfo;

typedef struct PackedVoxelGridInfo {
	int cloudSize;
	int gridSize;
	PackedVoxelGridCorner minCorner;
	PackedVoxelGridCorner maxCorner;
} PackedVoxelGridInfo;
