/**
 * Author:  Florian Stock, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2018
 * Author:  Leonardo Solis, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2018
 * License: Apache 2.0 (see attachached File)
 */
#ifndef DATATYPES_H
#define DATATYPES_H

#include <vector>

// 3d euclidean point with intensity
typedef struct PointXYZI {
    float data[4];
} PointXYZI;

typedef struct Matrix4f {
  float data[4][4];
} Matrix4f;

typedef struct Mat33 {
  #if defined (DOUBLE_FP)
  double data[3][3];
  #else
  float data[3][3];
  #endif
} Mat33;

typedef struct Mat66 {
  #if defined (DOUBLE_FP)
  double data[6][6];
  #else
  float data[6][6];
  #endif
} Mat66;

typedef struct Mat36 {
  #if defined (DOUBLE_FP)
  double data[3][6];
  #else
  float data[3][6];
  #endif
} Mat36;

typedef struct Mat186 {
  #if defined (DOUBLE_FP)
  double data[18][6];
  #else
  float data[18][6];
  #endif
} Mat186;


typedef struct Vec5 {
  #if defined (DOUBLE_FP)
  double data[5];
  #else
  float data[5];
  #endif
} Vec5;

#if defined (DOUBLE_FP)
typedef double Vec3[3];
#else
typedef float Vec3[3];
#endif

#if defined (DOUBLE_FP)
typedef double Vec6[6];
#else
typedef float Vec6[6];
#endif

typedef struct Point4d {
    float x,y,z,i;
} Point4d;

typedef std::vector<PointXYZI> PointCloudSource;
typedef PointCloudSource PointCloud;

typedef struct CallbackResult {
	bool converged;
	Matrix4f final_transformation;
	#if defined (DOUBLE_FP)
	double fitness_score;
	#else
	float fitness_score;
	#endif
} CallbackResult;

typedef struct Voxel {
	Mat33 invCovariance;
	Vec3 mean;
	int first;
} Voxel;

typedef struct PointVoxel {
	Mat33 invCovariance;
	Vec3 mean;
	int point;
} PointVoxel;


typedef std::vector<Voxel> VoxelGrid;

Matrix4f Matrix4f_Identity = {
	{{1.0, 0.0, 0.0, 0.0}, 
	 {0.0, 1.0, 0.0, 0.0},
	 {0.0, 0.0, 1.0, 0.0},
	 {0.0, 0.0, 0.0, 1.0}}
};

#define PI 3.1415926535897932384626433832795

#endif

