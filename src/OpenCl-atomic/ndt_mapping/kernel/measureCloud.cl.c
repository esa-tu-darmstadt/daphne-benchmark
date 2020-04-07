#ifndef RESOUTION
#define RESOLUTION 1
#endif
#ifndef INV_RESOLUTION
#define INV_RESOLUTION 1
#endif
#ifndef RADIUS
#define RADIUS 1
#endif
#ifndef RADIUS_FINAL
#define RADIUS_FINAL 1.001f
#endif

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define EPHOS_ATOMICS

typedef struct {
  double data[3][3];
} Mat33;

typedef double Vec3[3];

typedef struct {
    Mat33 invCovariance;
    Vec3 mean;
	int first;
} Voxel;

typedef struct {
	Mat33 invCovariance;
	Vec3 mean;
	int point;
} PointVoxel;


typedef struct {
    float data[4];
} PointXYZI;

typedef struct {
	float x;
	float y;
	float z;
	int iNext;
} PointQueue;

typedef struct VoxelGridInfo {
	int cloudSize;
	int gridSize;
	PointXYZI minVoxel;
	PointXYZI maxVoxel;
	int gridDimension[3];
} VoxelGridInfo;

typedef struct PackedGridCorner{
	int data[4];
} PackedGridCorner;

typedef struct PackedVoxelGridInfo {
	int cloudSize;
	int gridSize;
	PackedGridCorner minVoxel;
	PackedGridCorner maxVoxel;
	int gridDimension[3];
} PackedVoxelGridInfo;


typedef struct GridCorner {
	float data[3];
} GridCorner;

#ifdef EPHOS_ATOMICS

void slowAtomicFMin(__global float* fp, float fval) {
	volatile __global int* ip = (__global int*)fp;

	int ival = *((int*)&fval);
	int old;
	int iref;
	do {
		iref = atomic_or(ip, 0x0);
		float fref = *((float*)&iref);
		if (fval < fref) {
			old = atomic_cmpxchg(ip, iref, ival);
		} else {
			old = atomic_cmpxchg(ip, iref, iref);
		}
	} while (old != iref);

}

inline int fcomppack(float val) {
	int ival = as_int(val);
	return (-(ival & ~(0x1<<31)))*(ival>>31) + (ival*((ival>>31) ^ 0x1));
}
inline float fcompunpack(int val) {
	int ival = (-val | (0x1<<31))*(val>>31) + (val*((val>>31) ^ 0x1));
	return as_float(ival);
}
inline void atomic_gfmin(__global float* fp, float fval) {
	int ival = fcomppack(fval);
	__global int* ip = (__global int*)fp;
	atomic_min(ip, ival);
}
inline void atomic_gfmax(__global float* fp, float fval) {
	int ival = fcomppack(fval);
	__global int* ip = (__global int*)fp;
	atomic_max(ip, ival);
}
inline int pack_minmaxf(float val) {
	int ival = as_int(val);
	return (-(ival & ~(0x1<<31)))*(ival>>31) + (ival*((ival>>31) ^ 0x1));
}
inline float unpack_minmaxf(int val) {
	int ival = (-val | (0x1<<31))*(val>>31) + (val*((val>>31) ^ 0x1));
	return as_float(ival);
}
void slowAtomicFMax(volatile __global float* fp, float fval) {

	volatile __global int* ip = (__global int*)fp;

	int ival = *((int*)&fval);
	int old;
	int iref;
	do {
		iref = atomic_or(ip, 0x0);
		float fref = *((float*)&iref);
		if (fval > fref) {
			old = atomic_cmpxchg(ip, iref, ival);
		} else {
			old = atomic_cmpxchg(ip, iref, iref);
		}
	} while (old != iref);
}

__kernel void measureCloud(
	__global PackedVoxelGridInfo* restrict gridInfo,
	__global const PointXYZI* restrict pointCloud,
	__local PackedGridCorner* l_minimum,
	__local PackedGridCorner* l_maximum
) {
	// initialize local structures
	if (get_local_id(0) == 0) {
		*l_minimum = (PackedGridCorner){{ 0, 0, 0, 0 }};
		*l_maximum = (PackedGridCorner){{ 0, 0, 0, 0 }};
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	int iPoint = get_global_id(0);
	if (iPoint < gridInfo->cloudSize) {
		// compare with one point
		__global PointXYZI* point = &pointCloud[iPoint];
		for (int i = 0; i < 3; i++) {
			int packed = pack_minmaxf(point->data[i]);
			atomic_min(&gridInfo->minVoxel.data[i], packed);
			atomic_max(&gridInfo->maxVoxel.data[i], packed);
			//atomic_gfmin(&gridInfo->minVoxel.data[i], point->data[i]);
			//atomic_gfmax(&gridInfo->maxVoxel.data[i], point->data[i]);
		}
	}
	// update global measurement
	barrier(CLK_GLOBAL_MEM_FENCE);
	if (get_local_id(0) == 0) {

	}

}
#else

// atomicless variant
/**
 * Finds the point cloud extends for each work group.
 * input: point cloud to measure
 * input_size: number of cloud elements
 * gmins: low bounds, one entry for each work group
 * gmaxs: high bounds, one entry for each work group
 */
__kernel void measureCloud(
	__global const PointXYZI* restrict input,
	int input_size,
	__global PointXYZI* restrict gmins,
	__global PointXYZI* restrict gmaxs
) {
	// Indices
	int gid   = get_global_id(0);
	int lid   = get_local_id(0);
	int wgid  = get_group_id(0);
	int lsize = get_local_size(0);

	// Storing min/max values and in local memory
	__local PointXYZI lmins [NUMWORKITEMS_PER_WORKGROUP];
	__local PointXYZI lmaxs [NUMWORKITEMS_PER_WORKGROUP];

	// Storing min/max private variables
	float mymin [3];
	float mymax [3];

	// Initializing min/max values of local and private variables
	for (char n = 0; n < 3; n++) {
		lmins [lid].data[n] = INFINITY;
		lmaxs [lid].data[n] = - INFINITY;
		mymin[n] = INFINITY;
		mymax[n] = -INFINITY;
	}

	// # work-groups that execute this kernel
	int num_wg = get_num_groups(0); 

	// # elements (from which a min/max will be found) assigned to each work-group
	int num_elems_per_wg = /*input_size / num_wg*/ lsize;

	// Offsets 
	int offset_wg = num_elems_per_wg * wgid; // of each work-group
	int offset_wi = offset_wg + lid;         // of each work-item within a work-group

	// Iteration upper-bound for each wg
	int upper_bound = num_elems_per_wg * (wgid + 1);

	PointXYZI temp;
	if (offset_wi < input_size) {
		for (int i = offset_wi; i < upper_bound; i += lsize) {
			temp = input [i];
		for (char n = 0; n < 3; n++) {
		if (temp.data[n] < mymin[n]) {
			mymin[n] = temp.data[n];
		}

		if (temp.data[n] > mymax[n]) {
			mymax[n] = temp.data[n];
		}
			}
		}
		
		// Storing the min/max found by each work-item in local memory
	for (char n = 0; n < 3; n++) {
		lmins[lid].data[n] = mymin[n];
		lmaxs[lid].data[n] = mymax[n];
		}
	}
	// wait for the writes of work group scale to finish
	barrier(CLK_LOCAL_MEM_FENCE);
	// Finding the work-group min/max locally
	lsize = lsize >> 1; // binary reduction
	while (lsize > 0) {
		if (lid < lsize) {
			for (char n = 0; n < 3; n++) {
				if (lmins[lid].data[n] > lmins[lid+lsize].data[n]) {
					lmins[lid].data[n] = lmins[lid+lsize].data[n];
				}
				if (lmaxs[lid].data[n] < lmaxs[lid+lsize].data[n]) {
					lmaxs[lid].data[n] = lmaxs[lid+lsize].data[n];
				}
			}
		}
		lsize = lsize >> 1; // binary reduction
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	// Writing work-group minimum/maximum to global memory
	if (lid == 0) {
		gmins [wgid] = lmins[0];
		gmaxs [wgid] = lmaxs[0];
	}
}
#endif