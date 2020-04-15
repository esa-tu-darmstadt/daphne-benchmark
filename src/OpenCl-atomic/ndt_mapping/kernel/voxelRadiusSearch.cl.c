/**
 * Performs radius search on a voxel grid and multiple points at once.
 * In that each work item determines the near voxels of one point.
 * pointCloud: point cloud
 * voxelGrid: voxel grid
 * result: pairs of voxels and point indices
 * l_startIndex: start index of the work group in the result buffer
 * l_resultNo: number of entries written to the result buffer by a work group
 * pointNo: number of points in the point cloud
 * minVoxel: corner of the grid with minumum coordinates
 * maxVoxel: corner of the grid with maximum coordinates
 * voxelDimension_0: voxel grid size
 * voxelDimension_1: voxel grid size
 */
__kernel void radiusSearch(
	__global const VoxelGridInfo* restrict gridInfo,
	__global const PointXYZI* restrict pointCloud,
	__global const Voxel* restrict voxelGrid,
	__global PointVoxel* result,
	__global int* resultNo,
	__local int* l_startIndex,
	__local int* l_resultNo) {

	int iPoint = get_global_id(0);
	if (get_local_id(0) == 0) {
		*l_startIndex = -1;
		*l_resultNo = 0;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	int voxelBuffer[3*3*3];
	int voxelNo = 0;
	int iResult = -1;
	if (iPoint < gridInfo->cloudSize) {
		PointXYZI point = pointCloud[iPoint];
		// test all voxels in the vicinity
		for (float z = point.data[2] - RADIUS; z <= point.data[2] + RADIUS_FINAL; z += RESOLUTION) {
			for (float y = point.data[1] - RADIUS; y <= point.data[1] + RADIUS_FINAL; y += RESOLUTION) {
				for (float x = point.data[0] - RADIUS; x <= point.data[0] + RADIUS_FINAL; x += RESOLUTION) {
					// avoid accesses out of bounds
					if ((x < gridInfo->minCorner.data[0]) ||
						(x > gridInfo->maxCorner.data[0]) ||
						(y < gridInfo->minCorner.data[1]) ||
						(y > gridInfo->maxCorner.data[1]) ||
						(z < gridInfo->minCorner.data[2]) ||
						(z > gridInfo->maxCorner.data[2])) {
						// skip
					} else {
						// determine the distance to the voxel mean
						int iVoxel0 = (x - gridInfo->minCorner.data[0])*INV_RESOLUTION;
						int iVoxel1 = (y - gridInfo->minCorner.data[1])*INV_RESOLUTION;
						int iVoxel2 = (z - gridInfo->minCorner.data[2])*INV_RESOLUTION;
						int iVoxel = iVoxel0 + gridInfo->gridDimension.data[0]*(iVoxel1 + iVoxel2*gridInfo->gridDimension.data[1]);
						//int iVoxel =  linearizeCoord(x, y, z, minVoxel, voxelDimension_0, voxelDimension_1);
						float dx = voxelGrid[iVoxel].mean.data[0] - point.data[0];
						float dy = voxelGrid[iVoxel].mean.data[1] - point.data[1];
						float dz = voxelGrid[iVoxel].mean.data[2] - point.data[2];
						float dist = dx*dx + dy*dy + dz*dz;
						// add near cells to the results
						if (dist < RADIUS*RADIUS) {
							voxelBuffer[voxelNo] = iVoxel;
							voxelNo += 1;
						}
					}
				}
			}
		}
		if (voxelNo > 0) {
			iResult = atomic_add(l_resultNo, voxelNo);
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	if (get_local_id(0) == 0) {
		if (*l_resultNo > 0) {
			*l_startIndex = atomic_add(resultNo, *l_resultNo);
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	if (voxelNo > 0) {
		int startIndex = *l_startIndex + iResult;
		for (int i = 0; i < voxelNo; i++) {
			PointVoxel r = *((__global PointVoxel*)&voxelGrid[voxelBuffer[i]]);
			r.point = iPoint;
			result[startIndex + i] = r;
		}
	}
}