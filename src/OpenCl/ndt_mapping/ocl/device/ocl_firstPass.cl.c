/**
 * Author:  Leonardo Solis, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2018
 * License: Apache 2.0 (see attachached File)
 */
#if defined (DOUBLE_FP)
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

// Source: https://streamhpc.com/blog/2016-02-09/atomic-operations-for-floats-in-opencl-improved/
void atomicAdd_global_double(volatile __global double *addr, double val)
{
	union{
		ulong u64;
		double d64;
	} next, expected, current;

	current.d64 = *addr;

	do{
		expected.d64 = current.d64;
		next.d64 = expected.d64 + val;
		current.u64 = atom_cmpxchg( (volatile __global ulong *)addr, expected.u64, next.u64);
	} while( current.u64 != expected.u64 );
}
#else
void atomicAdd_global_singleFP(volatile __global float *addr, float val)
{
	union{
		uint u32;
		float f32;
	} next, expected, current;

	current.f32 = *addr;

	do{
		expected.f32 = current.f32;
		next.f32 = expected.f32 + val;
		current.u32 = atomic_cmpxchg( (volatile __global uint *)addr, expected.u32, next.u32);
	} while( current.u32 != expected.u32 );
}
#endif

/**
 * Reduces a multi dimensional voxel index to one dimension.
 */
inline int linearizeAddr(
	const int x, const int y, const int z,
	const int voxelDimension_0, const int voxelDimension_1)
{
	return  (x + voxelDimension_0 * (y + voxelDimension_1 * z));
}
/**
 * Reduces a continuous, multi dimensional coordinate inside a voxel grid to one dimension.
 */
inline int linearizeCoord(
	const float x, const float y, const float z, const PointXYZI minVoxel,
	const float inv_resolution, const int voxelDimension_0, const int voxelDimension_1)
{
	int idx_x = (x - minVoxel.data[0]) * inv_resolution;
	int idx_y = (y - minVoxel.data[1]) * inv_resolution;
	int idx_z = (z - minVoxel.data[2]) * inv_resolution;
	return linearizeAddr(idx_x, idx_y, idx_z, voxelDimension_0, voxelDimension_1);
}

/**
 * Assigns each point to its cell in a voxel grid.
 * input: point cloud
 * input_size: number of points in the cloud
 * targetcells: voxel grid
 * targetcells_size: number of cells in the voxel grid
 * minVoxel: voxel grid starting coordinates
 * inv_resolution: inverted cell distance
 * voxelDimension: multi dimensional voxel grid size
 */
__kernel
void __attribute__ ((reqd_work_group_size(NUMWORKITEMS_PER_WORKGROUP,1,1)))
firstPass(
	__global const PointXYZI* restrict input,
	int input_size,
	__global TargetGridLeadConstPtr* restrict targetcells,
	int targetcells_size,
	PointXYZI minVoxel,
	float inv_resolution,
	int voxelDimension_0,
	int voxelDimension_1)
{
	// Indices
	int gid = get_global_id(0);
	if (gid < input_size) {
		// Each work-item gets a different input target_ data element from global memory
		PointXYZI temp_target = input [gid];
		// index of the cell the point belongs to
		int voxelIndex = linearizeCoord (
			temp_target.data[0], temp_target.data[1], temp_target.data[2],
			minVoxel,
			inv_resolution,
			voxelDimension_0, voxelDimension_1
		);
		// increase the number of assigned points for normalization
		atomic_inc(&targetcells[voxelIndex].numberPoints);
		// calculating mean and covariance
		for (char row = 0; row < 3; row ++) {
			#if defined (DOUBLE_FP)
			atomicAdd_global_double(&targetcells[voxelIndex].mean[row], temp_target.data[row]);
			#else
			atomicAdd_global_singleFP(&targetcells[voxelIndex].mean[row], temp_target.data[row]);
			#endif
			for (char col = 0; col < 3; col ++) {
				#if defined (DOUBLE_FP)
				atomicAdd_global_double(&targetcells[voxelIndex].invCovariance.data[row][col], temp_target.data[row] * temp_target.data[col]);
				#else
				atomicAdd_global_singleFP(&targetcells[voxelIndex].invCovariance.data[row][col], temp_target.data[row] * temp_target.data[col]);
				#endif
			}
		}

	}
} 
