/**
 * Author:  Florian Stock, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2018
 * Author:  Leonardo Solis, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2018
 * License: Apache 2.0 (see attachached File)
 */
#ifndef EPHOS_DATATYPES_H
#define EPHOS_DATATYPES_H

#include "common/datatypes_base.h"

// #include <vector>
//
// // 3d euclidean point with intensity
// typedef struct PointXYZI {
//     float data[4];
// } PointXYZI;
//
// typedef struct Matrix4f {
//   float data[4][4];
// } Matrix4f;
//
// typedef struct Mat33 {
//   double data[3][3];
// } Mat33;
//
// typedef struct Mat66 {
//   double data[6][6];
// } Mat66;
//
// typedef struct Mat36 {
//   double data[3][6];
// } Mat36;
//
// typedef struct Mat186 {
//   double data[18][6];
// } Mat186;
//
//
// typedef struct Vec5 {
//   double data[5];
// } Vec5;
//
// typedef double Vec3[3];
//
// typedef double Vec6[6];
//
// typedef struct Point4d {
//     float x,y,z,i;
// } Point4d;
//
// typedef std::vector<PointXYZI> PointCloudSource;
// typedef PointCloudSource PointCloud;
//
// typedef struct CallbackResult {
// 	bool converged;
// 	std::vector<Matrix4f> intermediate_transformations;
// 	Matrix4f final_transformation;
// 	double fitness_score;
// } CallbackResult;
//
typedef struct VoxelMean {
	double data[3];
} VoxelMean;

typedef struct VoxelGridCorner {
	float data[3];
} VoxelGridCorner;

typedef struct PackedVoxelGridCorner {
	int data[3];
} PackedVoxelGridCorner;

typedef struct VoxelGridDimension {
	int data[2];
} VoxelGridDimension;

// typedef struct Voxel {
// 	Mat33 invCovariance;
// 	VoxelMean mean;
// 	int pointListBegin;
// #ifdef EPHOS_KERNEL_VOXEL_POINT_STORAGE
// 	int pointStorageLevel;
// 	PointXYZI pointStorage[EPHOS_KERNEL_VOXEL_POINT_STORAGE];
// #endif
// } Voxel;

typedef struct PointVoxel {
	Mat33 invCovariance;
	VoxelMean mean;
	int point;
} PointVoxel;

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

//typedef std::vector<Voxel> VoxelGrid;

#define PI 3.1415926535897932384626433832795

#endif

