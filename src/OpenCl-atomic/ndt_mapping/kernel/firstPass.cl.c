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
 * gridInfo: voxel grid info
 * voxelGrid: point cloud
 * voxelGrid: voxel grid
 */
__kernel void firstPass(
	__global VoxelGridInfo* restrict gridInfo,
	__global PointXYZI* restrict pointCloud,
	__global Voxel* restrict voxelGrid)
{
	// Indices
	int iPoint = get_global_id(0);
	if (iPoint < gridInfo->cloudSize) {
		// Each work-item gets a different input target_ data element from global memory
		PointXYZI point = pointCloud[iPoint];
		// index of the cell the point belongs to
		int iVoxel0 = (point.data[0] - gridInfo->minCorner.data[0])*INV_RESOLUTION;
		int iVoxel1 = (point.data[1] - gridInfo->minCorner.data[1])*INV_RESOLUTION;
		int iVoxel2 = (point.data[2] - gridInfo->minCorner.data[2])*INV_RESOLUTION;
		int iVoxel = iVoxel0 + gridInfo->gridDimension.data[0]*(iVoxel1 + gridInfo->gridDimension.data[1]*iVoxel2);
#ifdef EPHOS_VOXEL_POINT_STORAGE
		// insert into point storage if possible
		int iStorage = atomic_inc(&voxelGrid[iVoxel].pointStorageLevel);
		if (iStorage < EPHOS_VOXEL_POINT_STORAGE) {
			voxelGrid[iVoxel].pointStorage[iStorage] = point;
		} else {
			// append point as first element to list otherwise
			int next = atomic_xchg(&voxelGrid[iVoxel].pointListBegin, iPoint);
			// write next element to last vector component which is never used in kernels otherwise
			pointCloud[iPoint].data[3] = as_float(next);//*((float*)&next);
		}
#else // !EPHOS_VOXEL_POINT_STORAGE
		// append as first element to the list
		int next = atomic_xchg(&voxelGrid[iVoxel].pointListBegin, iPoint);
		// write next element to unused last vector component
		pointCloud[iPoint].data[3] = as_float(next);//*((float*)&next);
#endif // !EPHSO_VOXEL_POINT_STORAGE
	}
} 
