/**
 * Simple matrix inversion using the determinant
 */
void invertCovariance(VoxelCovariance* m)
{
	VoxelCovariance tmp;
	double det = 
		m->data[0][0] * (m->data[2][2] * m->data[1][1] - m->data[2][1] * m->data[1][2]) -
		m->data[1][0] * (m->data[2][2] * m->data[0][1] - m->data[2][1] * m->data[0][2]) +
		m->data[2][0] * (m->data[1][2] * m->data[0][1] - m->data[1][1] * m->data[0][2]);
	double invDet = 1.0 / det;

	// adjungated matrix of minors
	tmp.data[0][0] = m->data[2][2] * m->data[1][1] - m->data[2][1] * m->data[1][2];
	tmp.data[0][1] = -( m->data[2][2] * m->data[0][1] - m->data[2][1] * m->data[0][2]);
	tmp.data[0][2] = m->data[1][2] * m->data[0][1] - m->data[1][1] * m->data[0][2];

	tmp.data[1][0] = -( m->data[2][2] * m->data[0][1] - m->data[2][0] * m->data[1][2]);
	tmp.data[1][1] = m->data[2][2] * m->data[0][0] - m->data[2][1] * m->data[0][2];
	tmp.data[1][2] = -( m->data[1][2] * m->data[0][0] - m->data[1][0] * m->data[0][2]);

	tmp.data[2][0] = m->data[2][1] * m->data[1][0] - m->data[2][0] * m->data[1][1];
	tmp.data[2][1] = -( m->data[2][1] * m->data[0][0] - m->data[2][0] * m->data[0][1]);
	tmp.data[2][2] = m->data[1][1] * m->data[0][0] - m->data[1][0] * m->data[0][1];

	for (char row = 0; row < 3; row++)
		for (char col = 0; col < 3; col++)
			m->data[row][col] = tmp.data[row][col] * invDet;
}
/**
 * Normalizes a voxel grid after point assignment.
 * gridInfo: voxel grid info
 * voxelGrid: voxel grid
 * pointCloud: previously assigned points
 */
__kernel void secondPass(
	__global VoxelGridInfo* restrict gridInfo,
	__global Voxel* restrict voxelGrid,
	__global const PointXYZI* pointCloud)
{
	int iVoxel = get_global_id(0);
	if (iVoxel < gridInfo->gridSize) {
		Voxel voxel = voxelGrid[iVoxel];
		int pointNo = 0;
#ifdef EPHOS_VOXEL_POINT_STORAGE
		// collect points from the point storage
		for (int i = 0; i < EPHOS_VOXEL_POINT_STORAGE && i < voxel.pointStorageLevel; i++) {
			for (char row = 0; row < 3; row++) {
				//float r = voxel.pointStorage[i].data[row];
				voxel.mean.data[row] += voxel.pointStorage[i].data[row];
				for (char col = 0; col < 3; col++) {
					voxel.invCovariance.data[row][col] += voxel.pointStorage[i].data[row]*voxel.pointStorage[i].data[col];
				}
			}
			pointNo += 1;
		}
#endif // EPHOS_VOXEL_POINT_STORAGE
		// collect points from the point list
		int iNext = voxel.pointListBegin;
		while (iNext > -1) {
			PointXYZI point = pointCloud[iNext];
			for (char row = 0; row < 3; row ++) {
				voxel.mean.data[row] += point.data[row];
				for (char col = 0; col < 3; col ++) {
					voxel.invCovariance.data[row][col] += point.data[row]*point.data[col];
				}
			}
			// next queue item, interpret last component as int
			iNext = as_int(point.data[3]);//*((int*)&point.data[3]);
			pointNo += 1;
 		}
		// average the point sum
		double pointSum[3];
		for (char k = 0; k < 3; k++) {
			pointSum[k] = voxel.mean.data[k];
			voxel.mean.data[k] /= pointNo;
		}
		// postprocess the matrix
		for (char row = 0; row < 3; row++) {
			for (char col = 0; col < 3; col++) {
				double tmp = (voxel.invCovariance.data[row][col] - 2 *
					(pointSum[row] * voxel.mean.data[col])) /
 					gridInfo->gridSize + voxel.mean.data[row]*voxel.mean.data[col];
				voxel.invCovariance.data[row][col] =
					tmp * (gridInfo->gridSize - 1) / pointNo;
			}
		}
		invertCovariance(&voxel.invCovariance);
		voxelGrid[iVoxel] = voxel;
	}
}
