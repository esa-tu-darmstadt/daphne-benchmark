/**
 * Author:  Leonardo Solis, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2018
 * License: Apache 2.0 (see attachached File)
 */
/**
 * Simple matrix inversion using the determinant
 */
void invertMatrix(Mat33* m)
{
	Mat33 temp;
	#if defined (DOUBLE_FP)
	double det = 
		m->data[0][0] * (m->data[2][2] * m->data[1][1] - m->data[2][1] * m->data[1][2]) -
		m->data[1][0] * (m->data[2][2] * m->data[0][1] - m->data[2][1] * m->data[0][2]) +
		m->data[2][0] * (m->data[1][2] * m->data[0][1] - m->data[1][1] * m->data[0][2]);
	double invDet = 1.0 / det;
	#else
	float det = 
		m->data[0][0] * (m->data[2][2] * m->data[1][1] - m->data[2][1] * m->data[1][2]) -
		m->data[1][0] * (m->data[2][2] * m->data[0][1] - m->data[2][1] * m->data[0][2]) +
		m->data[2][0] * (m->data[1][2] * m->data[0][1] - m->data[1][1] * m->data[0][2]);
	float invDet = 1.0f / det;
	#endif

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
	__global TargetGridLeadConstPtr* restrict targetcells,
	int targetcells_size,
	int targetcells_size_minus_1)
{
	int gid = get_global_id(0);
	if (gid < targetcells_size) {
		TargetGridLeadConstPtr temp_tc = targetcells[gid];
		Vec3 pointSum;
		for (char k = 0; k < 3; k++) {
			pointSum[k] = temp_tc.mean[k];
			temp_tc.mean[k] /= temp_tc.numberPoints;
		}
		#if defined (DOUBLE_FP)
		double tmp;
		#else
		float tmp;
		#endif
		for (char row = 0; row < 3; row++) {
			for (char col = 0; col < 3; col++) {
				tmp = (temp_tc.invCovariance.data[row][col] - 2 * 
					(pointSum[row] * temp_tc.mean[col])) 
					/ targetcells_size + temp_tc.mean[row]*temp_tc.mean[col];
				temp_tc.invCovariance.data[row][col] = 
					tmp * (targetcells_size_minus_1) / temp_tc.numberPoints; 
			}
		}
		invertMatrix(&temp_tc.invCovariance);
		targetcells[gid] = temp_tc;
	}
}
