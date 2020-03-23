/**
 * Simple matrix inversion using the determinant
 */
void invertMatrix(Mat33* m)
{
	Mat33 temp;
	double det = 
		m->data[0][0] * (m->data[2][2] * m->data[1][1] - m->data[2][1] * m->data[1][2]) -
		m->data[1][0] * (m->data[2][2] * m->data[0][1] - m->data[2][1] * m->data[0][2]) +
		m->data[2][0] * (m->data[1][2] * m->data[0][1] - m->data[1][1] * m->data[0][2]);
	double invDet = 1.0 / det;

	// adjungated matrix of minors
	temp.data[0][0] = m->data[2][2] * m->data[1][1] - m->data[2][1] * m->data[1][2];
	temp.data[0][1] = -( m->data[2][2] * m->data[0][1] - m->data[2][1] * m->data[0][2]);
	temp.data[0][2] = m->data[1][2] * m->data[0][1] - m->data[1][1] * m->data[0][2];

	temp.data[1][0] = -( m->data[2][2] * m->data[0][1] - m->data[2][0] * m->data[1][2]);
	temp.data[1][1] = m->data[2][2] * m->data[0][0] - m->data[2][1] * m->data[0][2];
	temp.data[1][2] = -( m->data[1][2] * m->data[0][0] - m->data[1][0] * m->data[0][2]);

	temp.data[2][0] = m->data[2][1] * m->data[1][0] - m->data[2][0] * m->data[1][1];
	temp.data[2][1] = -( m->data[2][1] * m->data[0][0] - m->data[2][0] * m->data[0][1]);
	temp.data[2][2] = m->data[1][1] * m->data[0][0] - m->data[1][0] * m->data[0][1];

	for (char row = 0; row < 3; row++)
		for (char col = 0; col < 3; col++)
			m->data[row][col] = temp.data[row][col] * invDet;
}
/**
 * Normalizes a voxel grid after point assignment.
 * targetcells: voxel grid
 * targetcells_size: number of cells in the voxel grid
 * targetcells_size_minus_1: modified number of cells in the voxel grid
 */
__kernel
void __attribute__ ((reqd_work_group_size(NUMWORKITEMS_PER_WORKGROUP,1,1)))
secondPass(
	__global Voxel* restrict targetcells,
	__global const PointXYZI* input,
	int targetcells_size,
	int targetcells_size_minus_1)
{
	int gid = get_global_id(0);
	if (gid < targetcells_size) {
		Voxel temp_tc = targetcells[gid];
		int pointNo = 0;
		int iNext = temp_tc.first;
		while (iNext > -1) {
			PointXYZI temp_target = input[iNext];
			for (char row = 0; row < 3; row ++) {
				temp_tc.mean[row] += temp_target.data[row];
				for (char col = 0; col < 3; col ++) {
					temp_tc.invCovariance.data[row][col] += temp_target.data[row]*temp_target.data[col];
				}
			}
			// next queue item, interpret last component as int
			iNext = *((int*)&temp_target.data[3]);
			pointNo += 1;
 		}

		Vec3 pointSum;
		for (char k = 0; k < 3; k++) {
			pointSum[k] = temp_tc.mean[k];
			temp_tc.mean[k] /= pointNo;
		}
		double tmp;
		for (char row = 0; row < 3; row++) {
			for (char col = 0; col < 3; col++) {
				tmp = (temp_tc.invCovariance.data[row][col] - 2 * 
					(pointSum[row] * temp_tc.mean[col])) 
					/ targetcells_size + temp_tc.mean[row]*temp_tc.mean[col];
				temp_tc.invCovariance.data[row][col] = 
					tmp * (targetcells_size_minus_1) / pointNo;
			}
		}
		invertMatrix(&temp_tc.invCovariance);
		targetcells[gid] = temp_tc;
	}
}
