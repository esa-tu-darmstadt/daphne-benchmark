/**
 * Performs radius search on a voxel grid and multiple points at once.
 * In that each work item determines the near voxels of one point.
 * input: point cloud
 * target_cells: voxel grid
 * result: pairs of voxels and point indices
 * l_startIndex: start index of the work group in the result buffer
 * l_resultNo: number of entries written to the result buffer by a work group
 * pointNo: number of points in the input
 * minVoxel: corner of the grid with minumum coordinates
 * maxVoxel: corner of the grid with maximum coordinates
 * voxelDimension_0: voxel grid size
 * voxelDimension_1: voxel grid size
 */
__kernel
void __attribute__ ((reqd_work_group_size(NUMWORKITEMS_PER_WORKGROUP,1,1)))
radiusSearch(
	__global const PointXYZI* restrict input,
	__global const Voxel* restrict target_cells,
	__global PointVoxel* result,
	__global int* resultNo,
	__local int* l_startIndex,
	__local int* l_resultNo,
	int pointNo,
	PointXYZI minVoxel,
	PointXYZI maxVoxel,
	int voxelDimension_0,
	int voxelDimension_1) {

	int id = get_global_id(0);
	if (get_local_id(0) == 0) {
		*l_startIndex = -1;
		*l_resultNo = 0;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	int voxelBuffer[3*3*3];
	int voxelNo = 0;
	int iResult = -1;
	if (id < pointNo) {
		PointXYZI point = input[id];
		// test all voxels in the vicinity
		for (float z = point.data[2] - RADIUS; z <= point.data[2] + RADIUS_FINAL; z += RESOLUTION) {
			for (float y = point.data[1] - RADIUS; y <= point.data[1] + RADIUS_FINAL; y += RESOLUTION) {
				for (float x = point.data[0] - RADIUS; x <= point.data[0] + RADIUS_FINAL; x += RESOLUTION) {
					// avoid accesses out of bounds
					if ((x < minVoxel.data[0]) ||
						(x > maxVoxel.data[0]) ||
						(y < minVoxel.data[1]) ||
						(y > maxVoxel.data[1]) ||
						(z < minVoxel.data[2]) ||
						(z > maxVoxel.data[2])) {
						// skip
					} else {
						// determine the distance to the voxel mean
						int iVoxel =  linearizeCoord(x, y, z, minVoxel, voxelDimension_0, voxelDimension_1);
						float dx = target_cells[iVoxel].mean[0] - point.data[0];
						float dy = target_cells[iVoxel].mean[1] - point.data[1];
						float dz = target_cells[iVoxel].mean[2] - point.data[2];
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
			PointVoxel r = *((__global PointVoxel*)&target_cells[voxelBuffer[i]]);
			r.point = id;
			result[startIndex + i] = r;
		}
	}
}