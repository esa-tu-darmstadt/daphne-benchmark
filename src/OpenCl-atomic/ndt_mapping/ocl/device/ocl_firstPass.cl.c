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
	__global PointXYZI* restrict input,
	int input_size,
	__global Voxel* restrict targetcells,
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
		PointXYZI temp_target = input[gid];
		// index of the cell the point belongs to
		int voxelIndex = linearizeCoord (
			temp_target.data[0], temp_target.data[1], temp_target.data[2],
			minVoxel,
			inv_resolution,
			voxelDimension_0, voxelDimension_1
		);
		// append point to queue front
		int next = atomic_xchg(&targetcells[voxelIndex].first, gid);
		// write next element to last vector component (int bits written to float datum)
		input[gid].data[3] = *((float*)&next);
	}
} 
