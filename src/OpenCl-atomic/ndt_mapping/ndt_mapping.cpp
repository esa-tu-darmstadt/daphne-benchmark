/**
 * Author:  Florian Stock, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2018
 * Author:  Leonardo Solis, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2018
 * Author:  Thilo Gabel, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2019 - 2020
 * License: Apache 2.0 (see attached files)
 */

#include <cmath>
#include <iostream>
#include <sstream>
#include <fstream>
#include <limits>
#include <cstring>
#include <stdexcept>

#include "ndt_mapping.h"
#include "kernel/kernel.h"



ndt_mapping::ndt_mapping() : ndt_mapping_base(),
	computeEnv(),
	voxelGridBuffer(),
	pointCloudBuffer(),
	subvoxelBuffer(),
	counterBuffer(),
	gridInfoBuffer(),
#ifdef EPHOS_PINNED_MEMORY
	subvoxelHostBuffer(),
#elif defined(EPHOS_ZERO_COPY)
#else
	subvoxelStorage(nullptr),
#endif
	measureCloudKernel(),
	radiusSearchKernel(),
	initTargetCellsKernel(),
	firstPassKernel(),
	secondPassKernel(),
	maxComputeGridSize(0),
	maxComputeCloudSize(0) {
}

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
	// opencl setup
	try {
		std::vector<std::vector<std::string>> requiredExtensions = {
			{"cl_khr_fp64", "cl_amd_fp64"}
		};
		computeEnv = ComputeTools::find_compute_platform(EPHOS_PLATFORM_HINT_S, EPHOS_DEVICE_HINT_S,
			EPHOS_DEVICE_TYPE_S, requiredExtensions);
		std::cout << "OpenCL device: " << computeEnv.device.getInfo<CL_DEVICE_NAME>() << std::endl;
	} catch (std::logic_error& e) {
		std::cerr << e.what() << std::endl;
		exit(EXIT_FAILURE);
	}
	// Kernel code was stringified, rather than read from file
	std::string sourceCode = voxel_grid_ocl_kernel_source;

	//cl::Program::Sources sourcesCL = cl::Program::Sources(1, std::make_pair(sourceCode.c_str(), sourceCode.size()));
	cl::Program::Sources sourcesCL;
	std::vector<cl::Kernel> kernels;
	try {
		std::string sOptions =
#ifdef EPHOS_KERNEL_VOXEL_POINT_STORAGE
			" -DEPHOS_VOXEL_POINT_STORAGE=" STRINGIFY(EPHOS_KERNEL_VOXEL_POINT_STORAGE)
#else
			""
#endif
		;
		std::vector<std::string> kernelNames({
			"measureCloud",
			"initTargetCells",
			"firstPass",
			"secondPass",
			"radiusSearch"
		});
		cl::Program program = ComputeTools::build_program(computeEnv, sourceCode, sOptions,
			kernelNames, kernels);
	} catch (std::logic_error& e) {
		std::cerr << e.what() << std::endl;
		exit(EXIT_FAILURE);
	}
	measureCloudKernel = kernels[0];
	initTargetCellsKernel = kernels[1];
	firstPassKernel = kernels[2];
	secondPassKernel = kernels[3];
	radiusSearchKernel = kernels[4];
	std::cout << "done" << std::endl;
}
void ndt_mapping::quit() {
	ndt_mapping_base::quit();
#ifdef EPHOS_PINNED_MEMORY
	if (subvoxelStorage != nullptr) {
		computeEnv.cmdqueue.enqueueUnmapMemObject(subvoxelHostBuffer, subvoxelStorage);
		subvoxelStorage = nullptr;
	}
#elif defined(EPHOS_ZERO_COPY)
#else
	if (subvoxelStorage != nullptr) {
		delete[] subvoxelStorage;
		subvoxelStorage = nullptr;
	}
#endif
}

/**
 * Applies the transformation matrix to all point cloud elements
 * input: points to be transformed
 * output: transformed points
 * transform: transformation matrix
 */
void transformPointCloud(const PointCloud& input, PointCloud &output, Matrix4f transform);


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
	// TODO: test unused code here
	memset(&(hessian.data[0][0]), 0, sizeof(double) * 6 * 6);
	// move transformed cloud to device
	int pointNo = trans_cloud.size;
	computeEnv.cmdqueue.enqueueWriteBuffer(pointCloudBuffer, CL_FALSE, 0,
		sizeof(PointXYZI)*pointNo, trans_cloud.data);
	int nearVoxelNo = 0;
	computeEnv.cmdqueue.enqueueWriteBuffer(counterBuffer, CL_FALSE, 0, sizeof(int), &nearVoxelNo);
	computeEnv.cmdqueue.enqueueWriteBuffer(gridInfoBuffer, CL_FALSE, 0, sizeof(int), &pointNo);
	// call radius search kernel
	size_t local_size = EPHOS_KERNEL_WORK_GROUP_SIZE;
	size_t num_workgroups = pointNo/local_size + 1;
	size_t global_size = local_size*num_workgroups;
	computeEnv.cmdqueue.enqueueNDRangeKernel(
		radiusSearchKernel,
		cl::NDRange(0),
		cl::NDRange(global_size),
		cl::NDRange(local_size));
	// move near voxels to host
	computeEnv.cmdqueue.enqueueReadBuffer(counterBuffer, CL_TRUE, 0, sizeof(int), &nearVoxelNo);
	size_t nbytes_subvoxel = sizeof(PointVoxel)*nearVoxelNo;
	PointVoxel* storage_subvoxel = (PointVoxel*)computeEnv.cmdqueue.enqueueMapBuffer(subvoxelBuffer,
		CL_TRUE, CL_MAP_READ, 0, nbytes_subvoxel);
	// process near voxels
	for (int i = 0; i < nearVoxelNo; i++) {
		int iPoint = storage_subvoxel[i].point;
		PointXYZI& x_pt = input_cloud->data[iPoint];
		Vec3 x = {
			x_pt.data[0],
			x_pt.data[1],
			x_pt.data[2]
		};
		computePointDerivatives(x);
		VoxelMean* mean = &storage_subvoxel[i].mean;
		PointXYZI& x_trans_pt = trans_cloud.data[iPoint];
		Vec3 x_trans = {
			x_trans_pt.data[0] - mean->data[0],
			x_trans_pt.data[1] - mean->data[1],
			x_trans_pt.data[2] - mean->data[2]
		};
		Mat33& c_inv = storage_subvoxel[i].invCovariance;
		updateHessian(hessian, x_trans, c_inv);
	}
	computeEnv.cmdqueue.enqueueUnmapMemObject(subvoxelBuffer, storage_subvoxel);
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
	Vec6& score_gradient,
	Mat66& hessian,
	PointCloud& trans_cloud,
	Vec6& p,
	bool compute_hessian)
{
	memset(&(score_gradient[0]), 0, sizeof(double) * 6 );
	memset(&(hessian.data[0][0]), 0, sizeof(double) * 6 * 6);
	double score = 0.0;
	// Precompute Angular Derivatives (eq. 6.19 and 6.21)[Magnusson 2009]
	computeAngleDerivatives (p);
	// move transformed cloud to device
	int pointNo = trans_cloud.size;
#ifdef EPHOS_ZERO_COPY
	PointXYZI* pointCloudStorage = (PointXYZI*)computeEnv.cmdqueue.enqueueMapBuffer(pointCloudBuffer,
		CL_TRUE, CL_MAP_WRITE, 0, sizeof(PointXYZI)*pointNo);
	std::memcpy(pointCloudStorage, trans_cloud.data, sizeof(PointXYZI)*pointNo);
	computeEnv.cmdqueue.enqueueUnmapMemObject(pointCloudBuffer, pointCloudStorage);
#else
	computeEnv.cmdqueue.enqueueWriteBuffer(pointCloudBuffer, CL_FALSE,
		0, sizeof(PointXYZI)*pointNo, trans_cloud.data);
#endif
	int nearVoxelNo = 0;
	computeEnv.cmdqueue.enqueueWriteBuffer(counterBuffer, CL_FALSE, 0, sizeof(int), &nearVoxelNo);
	computeEnv.cmdqueue.enqueueWriteBuffer(gridInfoBuffer, CL_FALSE, 0, sizeof(int), &pointNo);
	// call radius search kernel
	//radiusSearchKernel.setArg(7, pointNo);
	size_t local_size = EPHOS_KERNEL_WORK_GROUP_SIZE;
	size_t num_workgroups = pointNo/local_size + 1;
	size_t global_size = local_size*num_workgroups;
	try {
	computeEnv.cmdqueue.enqueueNDRangeKernel(
		radiusSearchKernel,
		cl::NDRange(0),
		cl::NDRange(global_size),
		cl::NDRange(local_size));
	// move near voxels to host
	computeEnv.cmdqueue.enqueueReadBuffer(counterBuffer, CL_TRUE, 0, sizeof(int), &nearVoxelNo);
	} catch (cl::Error& e) {
		std::cout << e.what() << std::endl;
		exit(-2);
	}
#ifdef EPHOS_ZERO_COPY
	PointVoxel* subvoxelStorage = (PointVoxel*)computeEnv.cmdqueue.enqueueMapBuffer(subvoxelBuffer,
		CL_TRUE, CL_MAP_READ, 0, sizeof(PointVoxel)*nearVoxelNo);
#else
	computeEnv.cmdqueue.enqueueReadBuffer(subvoxelBuffer, CL_TRUE,
		0, sizeof(PointVoxel)*nearVoxelNo, subvoxelStorage);
#endif
	// process near voxels
	for (int i = 0; i < nearVoxelNo; i++) {
		int iPoint = subvoxelStorage[i].point;
		PointXYZI& x_pt = input_cloud->data[iPoint];
		Vec3 x = {
			x_pt.data[0],
			x_pt.data[1],
			x_pt.data[2]
		};
		computePointDerivatives(x);
		VoxelMean* mean = &subvoxelStorage[i].mean;
		PointXYZI& x_trans_pt = trans_cloud.data[iPoint];
		Vec3 x_trans = {
			x_trans_pt.data[0] - mean->data[0],
			x_trans_pt.data[1] - mean->data[1],
			x_trans_pt.data[2] - mean->data[2]
		};
		Mat33& c_inv = subvoxelStorage[i].invCovariance;
		score += updateDerivatives(score_gradient, hessian, x_trans, c_inv, compute_hessian);
	}
#ifdef EPHOS_ZERO_COPY
	computeEnv.cmdqueue.enqueueUnmapMemObject(subvoxelBuffer, subvoxelStorage);
#endif
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
inline float fcompunpack(int val) {
	int& ival = val;
	if (val>>31) {
		ival = -val;
		ival |= 0x1<<31;
	}
	return reinterpret_cast<float&>(ival);
}
inline int fcomppack(float val) {
	int& ival = reinterpret_cast<int&>(val);
	if ((ival>>31)) {
		return -(ival & (~(0x1<<31)));
	} else {
		return ival;
	}

}
int ndt_mapping::pack_minmaxf(float val) {
	int& ival = reinterpret_cast<int&>(val);
	return (-(ival & ~(0x1<<31)))*(ival>>31) + (ival*((ival>>31) ^ 0x1));
}
float ndt_mapping::unpack_minmaxf(int val) {
	int ival = (-val | (0x1<<31))*(val>>31) + (val*((val>>31) ^ 0x1));
	return reinterpret_cast<float&>(ival);
}

void ndt_mapping::initCompute()
{
	// create the point cloud buffers
	int pointNo = target_cloud->size;
	prepare_compute_buffers(pointNo, nullptr);
	PointXYZI minVoxel = target_cloud->data[0];
	PointXYZI maxVoxel = target_cloud->data[0];
	int voxelDimension[3];
	// move point cloud to device
#ifdef EPHOS_PINNED_MEMORY
	computeEnv.cmdqueue.enqueueWriteBuffer(pointCloudBuffer, CL_FALSE,
		0, sizeof(PointXYZI)*pointNo, target_cloud->data);
#elif defined(EPHOS_ZERO_COPY)
	PointXYZI* pointCloudStorage = (PointXYZI*)computeEnv.cmdqueue.enqueueMapBuffer(pointCloudBuffer,
		CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, 0, sizeof(PointXYZI)*pointNo);
	memcpy(pointCloudStorage, target_cloud->data, sizeof(PointXYZI)*pointNo);
	computeEnv.cmdqueue.enqueueUnmapMemObject(pointCloudBuffer, pointCloudStorage);
#else
	computeEnv.cmdqueue.enqueueWriteBuffer(pointCloudBuffer, CL_FALSE,
		0, sizeof(PointXYZI)*pointNo, target_cloud->data);
#endif

	// measure the given cloud
	size_t globalRange1 = pointNo;
	if (pointNo%EPHOS_KERNEL_WORK_GROUP_SIZE != 0) {
		size_t workgroupNo = pointNo/EPHOS_KERNEL_WORK_GROUP_SIZE + 1;
		globalRange1 = workgroupNo*EPHOS_KERNEL_WORK_GROUP_SIZE;
	}
#ifdef EPHOS_KERNEL_CLOUD_MEASURE
	int izero = pack_minmaxf(0.0f);
	PackedVoxelGridInfo gridInfo1 = {
		pointNo, // cloud size
		0, // grid size
		{ izero, izero, izero }, // min corner
		{ izero, izero, izero }, // max corner
	};
	computeEnv.cmdqueue.enqueueWriteBuffer(gridInfoBuffer, CL_FALSE,
		0, sizeof(PackedVoxelGridInfo), &gridInfo1);
	measureCloudKernel.setArg(0, gridInfoBuffer);
	measureCloudKernel.setArg(1, pointCloudBuffer);
	measureCloudKernel.setArg(2, cl::Local(sizeof(PointXYZI)));
	measureCloudKernel.setArg(3, cl::Local(sizeof(PointXYZI)));
//

	computeEnv.cmdqueue.enqueueNDRangeKernel(
		measureCloudKernel,
		cl::NDRange(0),
		cl::NDRange(globalRange1),
		cl::NDRange(EPHOS_KERNEL_WORK_GROUP_SIZE));

	// read back grid info and update host data
	computeEnv.cmdqueue.enqueueReadBuffer(gridInfoBuffer, CL_TRUE,
		0, sizeof(PackedVoxelGridInfo), &gridInfo1);
	for (int i = 0; i < 3; i++) {
		minVoxel.data[i] = unpack_minmaxf(gridInfo1.minCorner.data[i]) - transformation_epsilon_;
		maxVoxel.data[i] = unpack_minmaxf(gridInfo1.maxCorner.data[i]) + transformation_epsilon_;
		voxelDimension[i] = (maxVoxel.data[i] - minVoxel.data[i]) / resolution_ + 1;
	}

#else // !EPHOS_KERNEL_CLOUD_MEASURE
	for (int i = 1; i < pointNo; i++)
	{
		for (int elem = 0; elem < 3; elem++)
		{
			if ( target_cloud->data[i].data[elem] > maxVoxel.data[elem] )
				maxVoxel.data[elem] = target_cloud->data[i].data[elem];
			if ( target_cloud->data[i].data[elem] < minVoxel.data[elem] )
				minVoxel.data[elem] = target_cloud->data[i].data[elem];
		}
	}
	for (int i = 0; i < 3; i++) {
		minVoxel.data[i] = minVoxel.data[i] - transformation_epsilon_;
		maxVoxel.data[i] = maxVoxel.data[i] + transformation_epsilon_;
		voxelDimension[i] = (maxVoxel.data[i] - minVoxel.data[i]) / resolution_ + 1;
	}

#endif // !EPHOS_KERNEL_CLOUD_MEASURE
	// now resize the grid while other compute operations may still be active
	int cellNo = voxelDimension[0] * voxelDimension[1] * voxelDimension[2];
	VoxelGridInfo gridInfo2 = {
		pointNo, // cloudSize
		cellNo, // gridSize
		{ minVoxel.data[0], minVoxel.data[1], minVoxel.data[2] }, // min corner
		{ maxVoxel.data[0], maxVoxel.data[1], maxVoxel.data[2] }, // max corner
		{ voxelDimension[0], voxelDimension[1] } // grid dimension
	};
	// move updated grid info to device
	computeEnv.cmdqueue.enqueueWriteBuffer(gridInfoBuffer, CL_FALSE,
		0, sizeof(VoxelGridInfo), &gridInfo2);
	prepare_compute_buffers(0, voxelDimension);

	// call the grid initialization kernel
	initTargetCellsKernel.setArg(0, gridInfoBuffer);
	initTargetCellsKernel.setArg(1, voxelGridBuffer);
	//initTargetCellsKernel.setArg(1, cellNo);
	size_t globalRange2 = cellNo;
	if (cellNo%EPHOS_KERNEL_WORK_GROUP_SIZE != 0) {
		size_t workgroupNo = cellNo/EPHOS_KERNEL_WORK_GROUP_SIZE + 1;
		globalRange2 = workgroupNo*EPHOS_KERNEL_WORK_GROUP_SIZE;
	}
	computeEnv.cmdqueue.enqueueNDRangeKernel(
		initTargetCellsKernel,
		cl::NDRange(0),
		cl::NDRange(globalRange2),
		cl::NDRange(EPHOS_KERNEL_WORK_GROUP_SIZE));
	
	// call the kernel that assigns points to cells
	firstPassKernel.setArg(0, gridInfoBuffer);
	firstPassKernel.setArg(1, pointCloudBuffer);
	firstPassKernel.setArg(2, voxelGridBuffer);

	computeEnv.cmdqueue.enqueueNDRangeKernel(
		firstPassKernel,
		cl::NDRange(0),
		cl::NDRange(globalRange1),
		cl::NDRange(EPHOS_KERNEL_WORK_GROUP_SIZE));

	// call the kernel that postprocesses the voxel grid
	secondPassKernel.setArg(0, gridInfoBuffer);
	secondPassKernel.setArg(1, voxelGridBuffer);
	secondPassKernel.setArg(2, pointCloudBuffer);
	// ranges have been computed for the initialization kernel
	computeEnv.cmdqueue.enqueueNDRangeKernel(
		secondPassKernel,
		cl::NDRange(0),
		cl::NDRange(globalRange2),
		cl::NDRange(EPHOS_KERNEL_WORK_GROUP_SIZE));

	// the result will be used in the radius search kernel
	// prepare radius search kernel calls
	radiusSearchKernel.setArg(0, gridInfoBuffer);
	radiusSearchKernel.setArg(1, pointCloudBuffer);
	radiusSearchKernel.setArg(2, voxelGridBuffer);
	radiusSearchKernel.setArg(3, subvoxelBuffer);
	radiusSearchKernel.setArg(4, counterBuffer);
	radiusSearchKernel.setArg(5, cl::Local(sizeof(int)));
	radiusSearchKernel.setArg(6, cl::Local(sizeof(int)));
}
void ndt_mapping::cleanupCompute() {
	// nothing to do here
}
void ndt_mapping::prepare_compute_buffers(int cloudSize, int* gridSize) {
	// TODO: think about ways to increase sizes artificially as these increase with later test cases anyway
	// create buffers of satisfactory size
	if (maxComputeCloudSize == 0 && maxComputeGridSize == 0) {
		// create constant size buffers only once
		counterBuffer = cl::Buffer(computeEnv.context, CL_MEM_READ_WRITE,
			sizeof(int));
		gridInfoBuffer = cl::Buffer(computeEnv.context, CL_MEM_READ_WRITE,
			sizeof(VoxelGridInfo));
	}
	if (cloudSize > maxComputeCloudSize) {
		// resize buffers that depend on cloud size
#ifdef EPHOS_ZERO_COPY
		cl_mem_flags flags = CL_MEM_ALLOC_HOST_PTR;
#else
		cl_mem_flags flags = 0;
#endif
		pointCloudBuffer = cl::Buffer(computeEnv.context, flags | CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
			sizeof(PointXYZI)*cloudSize);
		subvoxelBuffer = cl::Buffer(computeEnv.context, flags | CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,
			sizeof(PointVoxel)*cloudSize);
		maxComputeCloudSize = cloudSize;
#ifdef EPHOS_PINNED_MEMORY
		if (subvoxelStorage != nullptr) {
			computeEnv.cmdqueue.enqueueUnmapMemObject(subvoxelHostBuffer, subvoxelStorage);
		}
		// TODO: since we have more operations here pinned memory might be considerably slower with current test data layout
		subvoxelHostBuffer = cl::Buffer(computeEnv.context,
			CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(PointVoxel)*cloudSize);
		subvoxelStorage = (PointVoxel*)computeEnv.cmdqueue.enqueueMapBuffer(subvoxelHostBuffer,
			CL_TRUE, CL_MAP_READ, 0, sizeof(PointVoxel)*cloudSize);
#elif defined(EPHOS_ZERO_COPY)
#else
		if (subvoxelStorage != nullptr) {
			delete[] subvoxelStorage;
		}
		subvoxelStorage = new PointVoxel[cloudSize];
#endif
	}
	if (gridSize != nullptr) {
		int voxelNo = gridSize[0]*gridSize[1]*gridSize[2];
		if (voxelNo > maxComputeGridSize) {
			// resize buffers that depend on grid size
			voxelGridBuffer = cl::Buffer(computeEnv.context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
				sizeof(Voxel)*voxelNo);
			maxComputeGridSize = voxelNo;
		}
	}
}

// create benchmark to run
ndt_mapping a;
benchmark& myKernel = a;
