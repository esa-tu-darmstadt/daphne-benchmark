/**
 * Author:  Florian Stock, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2018
 * Author:  Thilo Gabel, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2019
 * License: Apache 2.0 (see attached files)
 */


#include <cmath>
#include <iostream>
#include <sstream>
#include <fstream>
#include <limits>
#include <cstring>
#include <chrono>
#include <stdexcept>

#include "ndt_mapping.h"

#include "common/benchmark.h"


ndt_mapping::ndt_mapping() : ndt_mapping_base() {}
ndt_mapping::~ndt_mapping() {}

/**
 * Helper function to calculate the dot product of two vectors.
 */
double dot_product(Vec3 &a, Vec3 &b)
{
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/**
 * Helper function to calculate the dot product of two vectors.
 */
double dot_product6(Vec6 &a, Vec6 &b)
{
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3] + a[4] * b[4] + a[5] * b[5];
}

/**
 * Helper function for offset point generation.
 */
inline double auxilaryFunction_dPsiMT (double g_a, double g_0, double mu = 1.e-4) {
	return (g_a - mu * g_0);
}

/**
 * Helper function for difference offset generation.
 */
inline double auxilaryFunction_PsiMT (double a, double f_a, double f_0, double g_0, double mu = 1.e-4)
{
    return (f_a - f_0 - mu * g_0 * a);
}



void ndt_mapping::init() {
	std::cout << "init\n";
	ndt_mapping_base::init();
	std::cout << "done" << std::endl;
}
void ndt_mapping::quit() {
	ndt_mapping_base::quit();
}

int ndt_mapping::voxelRadiusSearch(VoxelGrid& grid, const PointXYZI& point,
	double radius, std::vector<Voxel*>& indices) {

	int result = 0;
	// make sure to search through all potentially near voxels
	float radiusFinal = radius + 0.01f;
	// test all voxels in the vicinity
	for (float z = point.data[2] - radius; z < point.data[2] + radiusFinal; z+= resolution_)
		for (float y = point.data[1] - radius; y < point.data[1] + radiusFinal; y+= resolution_)
			for (float x = point.data[0] - radius; x < point.data[0] + radiusFinal; x+= resolution_)
			{
				// avoid accesses out of bounds
// 				if ((x < minVoxel.data[0]) ||
// 					(x > maxVoxel.data[0]) ||
// 					(y < minVoxel.data[1]) ||
// 					(y > maxVoxel.data[1]) ||
// 					(z < minVoxel.data[2]) ||
// 					(z > maxVoxel.data[2])) {
//
// 					continue;
// 				}
				if ((x < grid.start[0]) ||
					(x > grid.start[0] + grid.dimension[0]) ||
					(y < grid.start[1]) ||
					(y > grid.start[1] + grid.dimension[1]) ||
					(z < grid.start[2]) ||
					(z > grid.start[2] + grid.dimension[2])) {

					continue;
				}
				// determine the distance to the voxel mean
				int iCellX = (x - grid.start[0])/resolution_;
				int iCellY = (y - grid.start[1])/resolution_;
				int iCellZ = (z - grid.start[2])/resolution_;
				int iCell = iCellX + grid.dimension[0]*(iCellY + iCellZ*grid.dimension[1]);
				//int iCell =  linearizeCoord(x, y, z);
				Voxel* cell = grid.data + iCell;
				if (cell->numberPoints > 0) {
					float dx = cell->mean[0] - point.data[0];
					float dy = cell->mean[1] - point.data[1];
					float dz = cell->mean[2] - point.data[2];
					float dist = dx * dx + dy * dy + dz * dz;
					// add near cells to the results
					if (dist < radius*radius)
					{
						result += 1;
						indices.push_back(cell);
					}
				}
			}
	return result;
}

double ndt_mapping::updateDerivatives (Vec6 &score_gradient,
				       Mat66 &hessian,
				       Vec3 &x_trans, Mat33 &c_inv,
				       bool compute_hessian)
{
	Vec3 cov_dxd_pi;

	// matrix preparation
	double xCx = c_inv.data[0][0] * x_trans[0] * x_trans[0] +
	c_inv.data[1][1] * x_trans[1] * x_trans[1] +
	c_inv.data[2][2] * x_trans[2] * x_trans[2] +
	(c_inv.data[0][1] + c_inv.data[1][0]) * x_trans[0] * x_trans[1] +
	(c_inv.data[0][2] + c_inv.data[2][0]) * x_trans[0] * x_trans[2] +
	(c_inv.data[1][2] + c_inv.data[2][1]) * x_trans[1] * x_trans[2];

	double e_x_cov_x = exp (-gauss_d2_ * (xCx) / 2);
	// Calculate probability of transtormed points existance, Equation 6.9 [Magnusson 2009]
	double score_inc = -gauss_d1_ * e_x_cov_x;
	e_x_cov_x = gauss_d2_ * e_x_cov_x;
	// Error checking for invalid values.
	if (e_x_cov_x > 1 || e_x_cov_x < 0 || e_x_cov_x != e_x_cov_x)
		return (0);
	// Reusable portion of Equation 6.12 and 6.13 [Magnusson 2009]
	e_x_cov_x *= gauss_d1_;
	for (int i = 0; i < 6; i++)
	{
		// Sigma_k^-1 d(T(x,p))/dpi, Reusable portion of Equation 6.12 and 6.13 [Magnusson 2009]
		//cov_dxd_pi = c_inv * point_gradient_.col (i);
		for (int row = 0; row < 3; row++)
		{
			cov_dxd_pi[row] = 0;
			for (int col = 0; col < 3; col++)
			cov_dxd_pi[row] += c_inv.data[row][col] * point_gradient_.data[col][i];
		}
		// Update gradient, Equation 6.12 [Magnusson 2009]
		score_gradient[i] += dot_product(x_trans, cov_dxd_pi) * e_x_cov_x;
		if (compute_hessian)
		{
			for (int j = 0; j < 6; j++)
			{
				Vec3 colVec = { point_gradient_.data[0][j], point_gradient_.data[1][j], point_gradient_.data[2][j] };
				Vec3 colVecHess = {colVec[0] + point_hessian_.data[3*i][j], colVec[1] + point_hessian_.data[3*i+1][j], colVec[2] + point_hessian_.data[3*i+2][j] };
				Vec3 matProd;
				for (int row = 0; row < 3; row++)
				{
					matProd[row] = 0;
					for (int col = 0; col < 3; col++)
					matProd[row] += c_inv.data[row][col] * colVecHess[col];
				}
				// Update hessian, Equation 6.13 [Magnusson 2009]
				hessian.data[i][j] += e_x_cov_x * (-gauss_d2_ * dot_product(x_trans, cov_dxd_pi) *
									dot_product(x_trans, matProd) +
									dot_product( colVec, cov_dxd_pi) );
			}
		}
	}

	return (score_inc);
}

void ndt_mapping::computePointDerivatives (Vec3 &x, bool compute_hessian)
{
	// Calculate first derivative of Transformation Equation 6.17 w.r.t. transform vector p.
	// Derivative w.r.t. ith element of transform vector corresponds to column i, Equation 6.18 and 6.19 [Magnusson 2009]
	point_gradient_.data[1][3] = dot_product(x, j_ang_a_);
	point_gradient_.data[2][3] = dot_product(x, j_ang_b_);
	point_gradient_.data[0][4] = dot_product(x, j_ang_c_);
	point_gradient_.data[1][4] = dot_product(x, j_ang_d_);
	point_gradient_.data[2][4] = dot_product(x, j_ang_e_);
	point_gradient_.data[0][5] = dot_product(x, j_ang_f_);
	point_gradient_.data[1][5] = dot_product(x, j_ang_g_);
	point_gradient_.data[2][5] = dot_product(x, j_ang_h_);

	if (compute_hessian)
	{
		//equation 6.21 [Magnusson 2009]
		Vec3 a, b, c, d, e, f;
		a[0] = 0;
		a[1] = dot_product(x, h_ang_a2_);
		a[2] = dot_product(x, h_ang_a3_);
		b[0] = 0;
		b[1] = dot_product(x, h_ang_b2_);
		b[2] = dot_product(x, h_ang_b3_);
		c[0] = 0;
		c[1] = dot_product(x, h_ang_c2_);
		c[2] = dot_product(x, h_ang_c3_);
		d[0] = dot_product(x, h_ang_d1_);
		d[1] = dot_product(x, h_ang_d2_);
		d[2] = dot_product(x, h_ang_d3_);
		e[0] = dot_product(x, h_ang_e1_);
		e[1] = dot_product(x, h_ang_e2_);
		e[2] = dot_product(x, h_ang_e3_);
		f[0] = dot_product(x, h_ang_f1_);
		f[1] = dot_product(x, h_ang_f2_);
		f[2] = dot_product(x, h_ang_f3_);
		// second derivative of Transformation Equation 6.17 w.r.t. transform vector p.
		// Derivative w.r.t. ith and jth elements of transform vector corresponds to the 3x1 block matrix starting at (3i,j), Equation 6.20 and 6.21 [Magnusson 2009]
		point_hessian_.data[9][3] = a[0];
		point_hessian_.data[10][3] = a[1];
		point_hessian_.data[11][3] = a[2];
		point_hessian_.data[12][3] = b[0];
		point_hessian_.data[13][3] = b[1];
		point_hessian_.data[14][3] = b[2];
		point_hessian_.data[15][3] = c[0];
		point_hessian_.data[16][3] = c[1];
		point_hessian_.data[17][3] = c[2];
		point_hessian_.data[9][4] = b[0];
		point_hessian_.data[10][4] = b[1];
		point_hessian_.data[11][4] = b[2];
		point_hessian_.data[12][4] = d[0];
		point_hessian_.data[13][4] = d[1];
		point_hessian_.data[14][4] = d[2];
		point_hessian_.data[15][4] = e[0];
		point_hessian_.data[16][4] = e[1];
		point_hessian_.data[17][4] = e[2];
		point_hessian_.data[9][5] = c[0];
		point_hessian_.data[10][5] = c[1];
		point_hessian_.data[11][5] = c[2];
		point_hessian_.data[12][5] = e[0];
		point_hessian_.data[13][5] = e[1];
		point_hessian_.data[14][5] = e[2];
		point_hessian_.data[15][5] = f[0];
		point_hessian_.data[16][5] = f[1];
		point_hessian_.data[17][5] = f[2];
	}
}

void ndt_mapping::computeHessian(
	Mat66 &hessian, PointCloud &trans_cloud, Vec6 &)
{
	// temporary data structures
	memset(&(hessian.data[0][0]), 0, sizeof(double) * 6 * 6);
	// Update hessian for each point, line 17 in Algorithm 2 [Magnusson 2009]
	int pointNo = input_cloud->size;
	for (size_t i = 0; i < pointNo; i++)
	{
		PointXYZI& x_trans_pt = trans_cloud.data[i];
		PointXYZI& x_pt = input_cloud->data[i];
		// Find neighbors
		std::vector<Voxel*> neighborhood;
		voxelRadiusSearch (target_grid, x_trans_pt, resolution_, neighborhood);
		// execute for each neighbor
		for (Voxel* cell : neighborhood)
		{
			// extract point
			Vec3 x = {
				x_pt.data[0],
				x_pt.data[1],
				x_pt.data[2]
			};
			Vec3 x_trans = {
				trans_cloud.data[i].data[0] - cell->mean[0],
				trans_cloud.data[i].data[1] - cell->mean[1],
				trans_cloud.data[i].data[2] - cell->mean[2]
			};
			Mat33 c_inv = cell->invCovariance;
			// Compute derivative of transform function w.r.t. transform vector, J_E and H_E in Equations 6.18 and 6.20 [Magnusson 2009]
			computePointDerivatives (x);
			// Update hessian, lines 21 in Algorithm 2, according to Equations 6.10, 6.12 and 6.13, respectively [Magnusson 2009]
			updateHessian (hessian, x_trans, c_inv);
		}
	}
}

void ndt_mapping::updateHessian (Mat66 &hessian, Vec3 &x_trans, Mat33 &c_inv)
{
	Vec3 cov_dxd_pi;
	// Equation 6.9 [Magnusson 2009]
	double xCx = c_inv.data[0][0] * x_trans[0] * x_trans[0] +
		c_inv.data[1][1] * x_trans[1] * x_trans[1] +
		c_inv.data[2][2] * x_trans[2] * x_trans[2] +
		(c_inv.data[0][1] + c_inv.data[1][0]) * x_trans[0] * x_trans[1] +
		(c_inv.data[0][2] + c_inv.data[2][0]) * x_trans[0] * x_trans[2] +
		(c_inv.data[1][2] + c_inv.data[2][1]) * x_trans[1] * x_trans[2];
	double e_x_cov_x = gauss_d2_ * exp (-gauss_d2_ * (xCx) / 2);

	// Error checking for invalid values.
	if (e_x_cov_x > 1 || e_x_cov_x < 0 || e_x_cov_x != e_x_cov_x)
		return;
	// Equation 6.12 and 6.13 [Magnusson 2009]
	e_x_cov_x *= gauss_d1_;

	for (int i = 0; i < 6; i++)
	{
		// Equation 6.12 and 6.13 [Magnusson 2009]
		for (int row = 0; row < 3; row++)
		{
			cov_dxd_pi[row] = 0;
			for (int col = 0; col < 3; col++)
			cov_dxd_pi[row] += c_inv.data[row][col] * point_gradient_.data[col][i];
		}
		
	for (int j = 0; j < 6; j++)
	{
		// Update hessian, Equation 6.13 [Magnusson 2009]
		Vec3 colVec = { point_gradient_.data[0][j], point_gradient_.data[1][j], point_gradient_.data[2][j] };
		Vec3 colVecHess = {colVec[0] + point_hessian_.data[3*i][j], colVec[1] + point_hessian_.data[3*i+1][j], colVec[2] + point_hessian_.data[3*i+2][j] };
		Vec3 matProd;
		for (int row = 0; row < 3; row++)
		{
			matProd[row] = 0;
			for (int col = 0; col < 3; col++)
				matProd[row] += c_inv.data[row][col] * colVecHess[col];
		}
		hessian.data[i][j] += e_x_cov_x * (-gauss_d2_ * dot_product(x_trans, cov_dxd_pi) *
							dot_product(x_trans, matProd) +
							dot_product( colVec, cov_dxd_pi) );
		}
	}
}

double ndt_mapping::computeDerivatives (
	Vec6 &score_gradient,
	Mat66 &hessian,
	PointCloud &trans_cloud,
	Vec6 &p,
	bool compute_hessian)
{
	// initialization to 0
	memset(&(score_gradient[0]), 0, sizeof(double) * 6 );
	memset(&(hessian.data[0][0]), 0, sizeof(double) * 6 * 6);
	double score = 0.0;
	// Precompute Angular Derivatives (eq. 6.19 and 6.21)[Magnusson 2009]
	computeAngleDerivatives (p);
	// Update gradient and hessian for each point, line 17 in Algorithm 2 [Magnusson 2009]
	for (size_t idx = 0; idx < input_cloud->size; idx++)
	{
		PointXYZI& x_trans_pt = trans_cloud.data[idx];
		PointXYZI& x_pt = input_cloud->data[idx];

		// Find nieghbors (Radius search has been experimentally faster than direct neighbor checking.
		std::vector<Voxel*> neighborhood;
		voxelRadiusSearch (target_grid, x_trans_pt, resolution_, neighborhood);

		for (Voxel* cell : neighborhood) {
			Vec3 x = {
				x_pt.data[0],
				x_pt.data[1],
				x_pt.data[2]
			};
			Vec3 x_trans = {
				x_trans_pt.data[0] - cell->mean[0],
				x_trans_pt.data[1] - cell->mean[1],
				x_trans_pt.data[2] - cell->mean[2]
			};
			// Uses precomputed covariance for speed.
			Mat33 c_inv = cell->invCovariance;
			// Equations 6.18 and 6.20 [Magnusson 2009]
			computePointDerivatives (x);
			// Equations 6.10, 6.12 and 6.13, respectively [Magnusson 2009]
			score += updateDerivatives (score_gradient, hessian, x_trans, c_inv, compute_hessian);
		}
	}
	return score;
}

void ndt_mapping::computeAngleDerivatives (Vec6 &p, bool compute_hessian)
{
	// Simplified math for near 0 angles
	double cx, cy, cz, sx, sy, sz;
	if (std::fabs (p[3]) < 10e-5)
	{
		//p(3) = 0;
		cx = 1.0;
		sx = 0.0;
	}
	else
	{
		cx = cos (p[3]);
		sx = sin (p[3]);
	}
	if (std::fabs (p[4]) < 10e-5)
	{
		//p(4) = 0;
		cy = 1.0;
		sy = 0.0;
	}
	else
	{
		cy = cos (p[4]);
		sy = sin (p[4]);
	}

	if (std::fabs (p[5]) < 10e-5)
	{
		//p(5) = 0;
		cz = 1.0;
		sz = 0.0;
	}
	else
	{
		cz = cos (p[5]);
		sz = sin (p[5]);
	}
	// Precomputed angular gradiant components. Letters correspond to Equation 6.19 [Magnusson 2009]
	j_ang_a_[0] = (-sx * sz + cx * sy * cz);
	j_ang_a_[1] = (-sx * cz -  cx * sy * sz);
	j_ang_a_[2] = (-cx * cy);
	j_ang_b_[0] = (cx * sz + sx * sy * cz);
	j_ang_b_[1] = (cx * cz - sx * sy * sz);
	j_ang_b_[2] = (-sx * cy);
	j_ang_c_[0] =  (-sy * cz);
	j_ang_c_[1] = sy * sz;
	j_ang_c_[2] = cy;
	j_ang_d_[0] = sx * cy * cz;
	j_ang_d_[1] = (-sx * cy * sz);
	j_ang_d_[2] = sx * sy;
	j_ang_e_[0] = (-cx * cy * cz);
	j_ang_e_[1] = cx * cy * sz;
	j_ang_e_[2] = (-cx * sy);
	j_ang_f_[0] = (-cy * sz);
	j_ang_f_[1] = (-cy * cz);
	j_ang_f_[2] = 0;
	j_ang_g_[0] = (cx * cz - sx * sy * sz);
	j_ang_g_[1] = (-cx * sz - sx * sy * cz);
	j_ang_g_[2] = 0;
	j_ang_h_[0] = (sx * cz + cx * sy * sz);
	j_ang_h_[1] =(cx * sy * cz - sx * sz);
	j_ang_h_[2] = 0;

	if (compute_hessian)
	{
		// Precomputed angular hessian components. Letters correspond to Equation 6.21 and numbers correspond to row index [Magnusson 2009]
		h_ang_a2_[0] = (-cx * sz - sx * sy * cz);
		h_ang_a2_[1] =  (-cx * cz + sx * sy * sz);
		h_ang_a2_[2] = sx * cy;
		h_ang_a3_[0] =  (-sx * sz + cx * sy * cz);
		h_ang_a3_[1] = (-cx * sy * sz - sx * cz);
		h_ang_a3_[2] = (-cx * cy);
		
		h_ang_b2_[0] = (cx * cy * cz);
		h_ang_b2_[1] = (-cx * cy * sz);
		h_ang_b2_[2] = (cx * sy);
		h_ang_b3_[0] = (sx * cy * cz);
		h_ang_b3_[1] = (-sx * cy * sz);
		h_ang_b3_[2] = (sx * sy);
		
		h_ang_c2_[0] = (-sx * cz - cx * sy * sz);
		h_ang_c2_[1] = (sx * sz - cx * sy * cz);
		h_ang_c2_[2] = 0;
		h_ang_c3_[0] = (cx * cz - sx * sy * sz);
		h_ang_c3_[1] = (-sx * sy * cz - cx * sz);
		h_ang_c3_[2] = 0;
		
		h_ang_d1_[0] = (-cy * cz);
		h_ang_d1_[1] = (cy * sz);
		h_ang_d1_[2] = (sy);
		h_ang_d2_[0] =  (-sx * sy * cz);
		h_ang_d2_[1] = (sx * sy * sz);
		h_ang_d2_[2] = (sx * cy);
		h_ang_d3_[0] = (cx * sy * cz);
		h_ang_d3_[1] = (-cx * sy * sz);
		h_ang_d3_[2] = (-cx * cy);
		
		h_ang_e1_[0] = (sy * sz);
		h_ang_e1_[1] = (sy * cz);
		h_ang_e1_[2] = 0;
		h_ang_e2_[0] =  (-sx * cy * sz);
		h_ang_e2_[1] = (-sx * cy * cz);
		h_ang_e2_[2] = 0;
		h_ang_e3_[0] = (cx * cy * sz);
		h_ang_e3_[1] = (cx * cy * cz);
		h_ang_e3_[2] = 0;
		
		h_ang_f1_[0] = (-cy * cz);
		h_ang_f1_[1] = (cy * sz);
		h_ang_f1_[2] = 0;
		h_ang_f2_[0] = (-cx * sz - sx * sy * cz);
		h_ang_f2_[1] = (-cx * cz + sx * sy * sz);
		h_ang_f2_[2] = 0;
		h_ang_f3_[0] = (-sx * sz + cx * sy * cz);
		h_ang_f3_[1] = (-cx * sy * sz - sx * cz);
		h_ang_f3_[2] = 0;
	}
}

/**
 * Helper function for simple matrix inversion using the determinant
 */
void invertMatrix(Mat33 &m)
{
	Mat33 temp;
	double det = m.data[0][0] * (m.data[2][2] * m.data[1][1] - m.data[2][1] * m.data[1][2]) -
	m.data[1][0] * (m.data[2][2] * m.data[0][1] - m.data[2][1] * m.data[0][2]) +
	m.data[2][0] * (m.data[1][2] * m.data[0][1] - m.data[1][1] * m.data[0][2]);
	double invDet = 1.0 / det;

	// adjungated matrix of minors
	temp.data[0][0] = m.data[2][2] * m.data[1][1] - m.data[2][1] * m.data[1][2];
	temp.data[0][1] = -( m.data[2][2] * m.data[0][1] - m.data[2][1] * m.data[0][2]);
	temp.data[0][2] = m.data[1][2] * m.data[0][1] - m.data[1][1] * m.data[0][2];

	temp.data[1][0] = -( m.data[2][2] * m.data[0][1] - m.data[2][0] * m.data[1][2]);
	temp.data[1][1] = m.data[2][2] * m.data[0][0] - m.data[2][1] * m.data[0][2];
	temp.data[1][2] = -( m.data[1][2] * m.data[0][0] - m.data[1][0] * m.data[0][2]);

	temp.data[2][0] = m.data[2][1] * m.data[1][0] - m.data[2][0] * m.data[1][1];
	temp.data[2][1] = -( m.data[2][1] * m.data[0][0] - m.data[2][0] * m.data[0][1]);
	temp.data[2][2] = m.data[1][1] * m.data[0][0] - m.data[1][0] * m.data[0][1];

	for (int row = 0; row < 3; row++)
	for (int col = 0; col < 3; col++)
		m.data[row][col] = temp.data[row][col] * invDet;

}

void ndt_mapping::initCompute()
{
	// measure the cloud

	PointXYZI minVoxel = target_cloud->data[0];
	PointXYZI maxVoxel = minVoxel;
	//int voxelDimension[3];

	int pointNo = target_cloud->size;

	for (int i = 1; i < pointNo; i++)
	{
		PointXYZI* point = target_cloud->data + i;
		for (int elem = 0; elem < 3; elem++)
		{
			if (point->data[elem] > maxVoxel.data[elem]) {
				maxVoxel.data[elem] = point->data[elem];
			}
			if (point->data[elem] < minVoxel.data[elem]) {
				minVoxel.data[elem] = point->data[elem];
			}
		}
	}
	// span over point cloud
	for (int i = 0; i < 3; i++) {
		//minVoxel.data[i] -= resolution_*0.5f;
		minVoxel.data[i] -= 0.01f;
		maxVoxel.data[i] += 0.01f;
		//voxelDimension[i] = (maxVoxel.data[i] - minVoxel.data[i])/resolution_ + 1;
		target_grid.dimension[i] = (maxVoxel.data[i] - minVoxel.data[i])/resolution_ + 1;
		target_grid.start[i] = minVoxel.data[i];
	}

	// initialize the voxel grid
	// spans over the point cloud
	int cellNo = target_grid.dimension[0]*target_grid.dimension[1]*target_grid.dimension[2];
	//voxelDimension[0]*voxelDimension[1]*voxelDimension[2];
	target_grid.data = new Voxel[cellNo];


	for (int i = 0; i < cellNo; i++)
	{
		target_grid.data[i] = (Voxel){
			{ 0.0, 0.0, 1.0,
			  0.0, 1.0, 0.0,
			  1.0, 0.0, 0.0 },
			{ 0.0, 0.0, 0.0 },
			0
		};
	}

	// assign the points to their respective voxel
	for (int i = 0; i < pointNo; i++)
	{
		PointXYZI* point = target_cloud->data + i;
		//int iVoxel = linearizeCoord( point->data[0], point->data[1], point->data[2]);
		int iCellX = (point->data[0] - target_grid.start[0])/resolution_;
		int iCellY = (point->data[1] - target_grid.start[1])/resolution_;
		int iCellZ = (point->data[2] - target_grid.start[2])/resolution_;
		int iCell = iCellX + target_grid.dimension[0]*(iCellY + iCellZ*target_grid.dimension[1]);
		Voxel* cell = target_grid.data + iCell;

		cell->mean[0] += point->data[0];
		cell->mean[1] += point->data[1];
		cell->mean[2] += point->data[2];
		cell->numberPoints += 1;
		// sum up for single pass covariance calculation
		for (int row = 0; row < 3; row ++)
		for (int col = 0; col < 3; col ++)
			cell->invCovariance.data[row][col] += point->data[row]*point->data[col];
	}
	// finish the voxel grid
	// perform normalization
	for (int i = 0; i < cellNo; i++)
	{
		Voxel* cell = target_grid.data + i;
		if (cell->numberPoints > 2) {
			Vec3 pointSum = {cell->mean[0], cell->mean[1], cell->mean[2]};
			double invPointNo = 1.0/cell->numberPoints;
			cell->mean[0] *= invPointNo;
			cell->mean[1] *= invPointNo;
			cell->mean[2] *= invPointNo;
			// complete the inverted covariance matrix
			for (int row = 0; row < 3; row++)
				for (int col = 0; col < 3; col++)
				{
					double cov = (cell->invCovariance.data[row][col] -
						2*(pointSum[row] * cell->mean[col]))/cell->numberPoints +
						cell->mean[row]*cell->mean[col];
					cell->invCovariance.data[row][col] =
						cov*(cell->numberPoints - 1.0)/cell->numberPoints;
				}
			invertMatrix(cell->invCovariance);
		} else {
			// cells with point number 0 will get sorted out
			// the rest of the cell content does not matter
			cell->numberPoints = 0;
		}
	}
}
void ndt_mapping::cleanupCompute() {
	delete[] target_grid.data;
	target_grid.data = nullptr;
}

// create benchmark to execute
ndt_mapping a;
benchmark& myKernel = a;
