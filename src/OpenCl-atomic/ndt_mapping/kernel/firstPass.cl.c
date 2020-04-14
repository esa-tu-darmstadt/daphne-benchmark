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
	const int voxelDimension_0, const int voxelDimension_1)
{
	int idx_x = (x - minVoxel.data[0]) * INV_RESOLUTION;
	int idx_y = (y - minVoxel.data[1]) * INV_RESOLUTION;
	int idx_z = (z - minVoxel.data[2]) * INV_RESOLUTION;
	return linearizeAddr(idx_x, idx_y, idx_z, voxelDimension_0, voxelDimension_1);
}

/**
 * Assigns each point to its cell in a voxel grid.
 * input: point cloud
 * input_size: number of points in the cloud
 * targetcells: voxel grid
 * targetcells_size: number of cells in the voxel grid
 * minVoxel: voxel grid starting coordinates
 * voxelDimension: multi dimensional voxel grid size
 */
//__kernel
//void __attribute__ ((reqd_work_group_size(NUMWORKITEMS_PER_WORKGROUP,1,1)))
//firstPass(
__kernel void firstPass(
	__global PointXYZI* restrict input,
	int input_size,
	__global Voxel* restrict targetcells,
	int targetcells_size,
	PointXYZI minVoxel,
	int voxelDimension_0,
	int voxelDimension_1)
{
	// Indices
	int iPoint = get_global_id(0);
	if (iPoint < input_size) {
		// Each work-item gets a different input target_ data element from global memory
		PointXYZI point = input[iPoint];
		// index of the cell the point belongs to
		int voxelIndex = linearizeCoord (
			point.data[0], point.data[1], point.data[2],
			minVoxel,
			voxelDimension_0, voxelDimension_1
		);
#ifdef EPHOS_VOXEL_POINT_STORAGE
		// insert into point storage if possible
		int iStorage = atomic_inc(&targetcells[voxelIndex].pointStorageLevel);
		if (iStorage < EPHOS_VOXEL_POINT_STORAGE) {
			targetcells[voxelIndex].pointStorage[iStorage] = point;
		} else {
			// append point as first element to list otherwise
			int next = atomic_xchg(&targetcells[voxelIndex].pointListBegin, iPoint);
			// write next element to last vector component which is never used in kernels otherwise
			input[iPoint].data[3] = as_float(next);//*((float*)&next);
			//input[iPoint].data[3] = (&next);
		}
#else // !EPHOS_VOXEL_POINT_STORAGE
		// append as first element to the list
		int next = atomic_xchg(&targetcells[voxelIndex].pointListBegin, iPoint);
		// write next element to unused last vector component
		input[iPoint].data[3] = as_float(next);//*((float*)&next);
#endif // !EPHSO_VOXEL_POINT_STORAGE
	}
} 
