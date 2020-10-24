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

typedef struct {
    float data[4];
} PointXYZI;

typedef struct {
    Mat33 invCovariance;
    double mean[3];
	int pointListBegin;
#ifdef EPHOS_VOXEL_POINT_STORAGE
	int pointStorageLevel;
	PointXYZI pointStorage[EPHOS_VOXEL_POINT_STORAGE];
#endif
} Voxel;

typedef struct {
	Mat33 invCovariance;
	double mean[3];
	int point;
} PointVoxel;

typedef struct {
	float x;
	float y;
	float z;
	int iNext;
} PointQueue;


// typedef struct PackedVoxelGridCorner{
// 	int data[3];
// } PackedVoxelGridCorner;

typedef int PackedVoxelGridCorner[3];


typedef struct VoxelGridInfo {
	int cloudSize;
	int gridSize;
	float minCorner[3];
	int gridDimension[3];
} VoxelGridInfo;

typedef struct PackedVoxelGridInfo {
	int cloudSize;
	int gridSize;
	int minCorner[3];
	int maxCorner[3];
} PackedVoxelGridInfo;
