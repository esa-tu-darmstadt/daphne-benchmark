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

inline int pack_minmaxf(float val) {
	int ival = as_int(val);
	return (-(ival & ~(0x1<<31)))*(ival>>31) + (ival*((ival>>31) ^ 0x1));
}
inline float unpack_minmaxf(int val) {
	int ival = (-val | (0x1<<31))*(val>>31) + (val*((val>>31) ^ 0x1));
	return as_float(ival);
}

__kernel void measureCloud(
	__global PackedVoxelGridInfo* restrict gridInfo,
	__global const PointXYZI* restrict pointCloud,
	__local PackedVoxelGridCorner* l_minimum,
	__local PackedVoxelGridCorner* l_maximum
) {
	// initialize local structures
	if (get_local_id(0) == 0) {
		*l_minimum = (PackedVoxelGridCorner){{ 0.0f, 0.0f, 0.0f }};
		*l_maximum = (PackedVoxelGridCorner){{ 0.0f, 0.0f, 0.0f }};
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	int iPoint = get_global_id(0);
	if (iPoint < gridInfo->cloudSize) {
		// compare with one point
		// build local corners
		__global const PointXYZI* point = &pointCloud[iPoint];
		int packed0 = pack_minmaxf(point->data[0]);
		atomic_min(&l_minimum->data[0], packed0);
		int packed1 = pack_minmaxf(point->data[1]);
		atomic_min(&l_minimum->data[1], packed1);
		int packed2 = pack_minmaxf(point->data[2]);
		atomic_min(&l_minimum->data[2], packed2);

		atomic_max(&l_maximum->data[0], packed0);
		atomic_max(&l_maximum->data[1], packed1);
		atomic_max(&l_maximum->data[2], packed2);
	}
	// update global measurement
	barrier(CLK_GLOBAL_MEM_FENCE);
	if (get_local_id(0) == 0) {
		// build global corners with one element of the work group
		atomic_min(&gridInfo->minCorner.data[0], (*l_minimum).data[0]);
		atomic_min(&gridInfo->minCorner.data[1], (*l_minimum).data[1]);
		atomic_min(&gridInfo->minCorner.data[2], (*l_minimum).data[2]);

		atomic_max(&gridInfo->maxCorner.data[0], (*l_maximum).data[0]);
		atomic_max(&gridInfo->maxCorner.data[1], (*l_maximum).data[1]);
		atomic_max(&gridInfo->maxCorner.data[2], (*l_maximum).data[2]);
	}

}
