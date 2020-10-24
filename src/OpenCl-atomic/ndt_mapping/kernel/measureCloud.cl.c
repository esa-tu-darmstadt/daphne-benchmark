/**
 * Author:  Thilo Gabel, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2020
 * License: Apache 2.0 (see attachached File)
 */

/**
 * Packs a floating point value to an integer representation
 * that has the same comparison semantics as the original float.
 * val: value to pack
 * return: packed value
 */
inline int pack_minmaxf(float val) {
	int ival = as_int(val);
	return (-(ival & ~(0x1<<31)))*(ival>>31) + (ival*((ival>>31) ^ 0x1));
}
/**
 * Brings a packed floating point value into floating point representation.
 * val: value to unpack
 * return: unpacked value
 */
inline float unpack_minmaxf(int val) {
	int ival = (-val | (0x1<<31))*(val>>31) + (val*((val>>31) ^ 0x1));
	return as_float(ival);
}
/**
 * Determines the two significant rectangular corners of a point cloud.
 * Utilizes packed positional data.
 * gridInfo: preliminarily filled point cloud info
 * pointCloud: the points to measure
 * l_minimum: local lower value corner
 * l_maximum: local higher value corner
 */
__kernel void measureCloud(
	__global PackedVoxelGridInfo* restrict gridInfo,
	__global const PointXYZI* restrict pointCloud,
	__local int* l_minimum,
	__local int* l_maximum
) {
	// initialize local structures
	if (get_local_id(0) == 0) {
		l_minimum[0] = as_int(0.0f);// (PackedVoxelGridCorner){ 0.0f, 0.0f, 0.0f };
		l_minimum[1] = as_int(0.0f);
		l_minimum[2] = as_int(0.0f);
		l_maximum[0] = as_int(0.0f);
		l_maximum[1] = as_int(0.0f);
		l_maximum[2] = as_int(0.0f);
		//(*l_maximum)[0] = (PackedVoxelGridCorner){ 0.0f, 0.0f, 0.0f };
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	int iPoint = get_global_id(0);
	if (iPoint < gridInfo->cloudSize) {
		// compare with one point
		// build local corners
		__global const PointXYZI* point = &pointCloud[iPoint];
		int packed0 = pack_minmaxf(point->data[0]);
		atomic_min(&l_minimum[0], packed0);
		int packed1 = pack_minmaxf(point->data[1]);
		atomic_min(&l_minimum[1], packed1);
		int packed2 = pack_minmaxf(point->data[2]);
		atomic_min(&l_minimum[2], packed2);

		atomic_max(&l_maximum[0], packed0);
		atomic_max(&l_maximum[1], packed1);
		atomic_max(&l_maximum[2], packed2);
	}
	// update global measurement
	barrier(CLK_GLOBAL_MEM_FENCE);
	if (get_local_id(0) == 0) {
		// build global corners with one element of the work group
		atomic_min(&gridInfo->minCorner[0], l_minimum[0]);
		atomic_min(&gridInfo->minCorner[1], l_minimum[1]);
		atomic_min(&gridInfo->minCorner[2], l_minimum[2]);

		atomic_max(&gridInfo->maxCorner[0], l_maximum[0]);
		atomic_max(&gridInfo->maxCorner[1], l_maximum[1]);
		atomic_max(&gridInfo->maxCorner[2], l_maximum[2]);
	}

}
