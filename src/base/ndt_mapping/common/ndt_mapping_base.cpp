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
#include <cassert>

#include "ndt_mapping_base.h"
#include "common/benchmark.h"



ndt_mapping_base::ndt_mapping_base() :
	benchmark(),
	read_testcases(0),
	input_file(),
	output_file(),
	datagen_file(),
	error_so_far(false), max_delta(0.0),
	outlier_ratio_(0.55), resolution_(1.0), trans_eps_(0.01), step_size_(0.1),
	// TODO: check whether this is supposed to be 0 or 30
	iter(30), max_iterations_(30),
	previous_transformation_(), transformation_(), final_transformation_(),
	intermediate_transformations_(),
	gauss_d1_(0.0), gauss_d2_(0.0),
	point_gradient_(),
	point_hessian_(),
	h_ang_a2_(), h_ang_a3_(),
	h_ang_b2_(), h_ang_b3_(),
	h_ang_c2_(), h_ang_c3_(),
	h_ang_d1_(), h_ang_d2_(), h_ang_d3_(),
	h_ang_e1_(), h_ang_e2_(), h_ang_e3_(),
	h_ang_f1_(), h_ang_f2_(), h_ang_f3_(),
	j_ang_a_(), j_ang_b_(), j_ang_c_(), j_ang_d_(), j_ang_e_(), j_ang_f_(), j_ang_g_(), j_ang_h_(),
	transformation_probability_(0.0), transformation_epsilon_(0.1),
	filtered_scan(), maps(), init_guess(), results(), grids(),
	input_cloud(nullptr), target_cloud(nullptr), target_grid() {
	//,
	//minVoxel(), maxVoxel(), voxelDimension() {
}
ndt_mapping_base::~ndt_mapping_base() {}

// int ndt_mapping_base::linearizeAddr(const int x, const int y, const int z)
// {
// 	return  (x + voxelDimension[0] * (y + voxelDimension[1] * z));
// }
//
// int ndt_mapping_base::linearizeCoord(const float x, const float y, const float z)
// {
// 	int idx_x = (x - minVoxel.data[0]) / resolution_;
// 	int idx_y = (y - minVoxel.data[1]) / resolution_;
// 	int idx_z = (z - minVoxel.data[2]) / resolution_;
// 	return linearizeAddr(idx_x, idx_y, idx_z);
// }

// int ndt_mapping_base::linearizeCoord(const float x, const float y, const float z) {
// 	int idx_x = (x - minVoxel.data[0]) / resolution_;
// 	int idx_y = (y - minVoxel.data[1]) / resolution_;
// 	int idx_z = (z - minVoxel.data[2]) / resolution_;
// 	return linearizeAddr(idx_x, idx_y, idx_z);
// }
/**
 * Helper function to calculate the dot product of two vectors.
 */
double ndt_mapping_base::dot_product(Vec3 &a, Vec3 &b)
{
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/**
 * Helper function to calculate the dot product of two vectors.
 */
double ndt_mapping_base::dot_product6(Vec6 &a, Vec6 &b)
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

/**
 * Solves Ax = b for x.
 * Maybe not as good when handling very ill conditioned systems, but is faster for a 6x6 matrix
 * and works well enough in practice.
 */
void solve(Vec6& result, Mat66 A, Vec6& b)
{
	double pivot;

	// bring to upper diagonal
	for(int j = 0; j < 6; j++)
	{
		double max = fabs(A.data[j][j]);
		int mi = j;
		for (int i = j + 1; i < 6; i++)
			if (fabs(A.data[i][j]) > max)
			{
				mi = i;
				max = fabs(A.data[i][j]);
			}
		// swap lines mi and j
		if (mi !=j)
			for (int i = 0; i < 6; i++)
			{
				double temp = A.data[mi][i];
				A.data[mi][i] = A.data[j][i];
				A.data[j][i] = temp;
			}
		if (max == 0.0) {
			// singular matrix
			A.data[j][j] = EPHOS_MAX_TRANSLATION_EPS;
		}
		// subtract lines to yield a triagonal matrix
		for (int i = j+1; i < 6; i++)
		{
			pivot=A.data[i][j]/A.data[j][j];
			for(int k = 0; k < 6; k++)
			{
				A.data[i][k]=A.data[i][k]-pivot*A.data[j][k];
			}
			b[i]=b[i]-pivot*b[j];
		}
	}
	// backward substituion
	result[5]=b[5]/A.data[5][5];
	for( int i = 4; i >= 0; i--)
	{
		double sum=0.0;
		for(int j = i+1; j < 6; j++)
		{
			sum=sum+A.data[i][j]*result[j];
		}
		result[i]=(b[i]-sum)/A.data[i][i];
	}
}

void ndt_mapping_base::init() {
	// open data file streams
	input_file.exceptions ( std::ifstream::failbit | std::ifstream::badbit );
	output_file.exceptions ( std::ifstream::failbit | std::ifstream::badbit );
	try {
		input_file.open("../../../data/ndt_input.dat", std::ios::binary);
	} catch (std::ifstream::failure& e) {
		std::cerr << "Error opening the testcase file" << std::endl;
		exit(-3);
	}
	try {
		output_file.open("../../../data/ndt_output.dat", std::ios::binary);
	}  catch (std::ifstream::failure& e) {
		std::cerr << "Error opening the results file" << std::endl;
		exit(-3);
	}
#ifdef EPHOS_TESTDATA_GEN
	try {
		datagen_file.open("../../../data/ndt_output_gen.dat", std::ios::binary);
	} catch (std::ofstream::failure& e) {
		std::cerr << "Error opening the datagen file" << std::endl;
		exit(-3);
	}
#endif
	// consume the number of testcases from the testcase file
	try {
		testcases = read_testdata_signature(input_file, output_file);
	} catch (std::ifstream::failure& e) {
		std::cerr << e.what() << std::endl;
		exit(-3);
	}
#ifdef EPHOS_TESTCASE_LIMIT
	if (EPHOS_TESTCASE_LIMIT < testcases) {
		testcases = EPHOS_TESTCASE_LIMIT;
	}
#endif
	// prepare the first iteration
	error_so_far = false;
	max_delta = 0.0;
	input_cloud = nullptr;
	target_cloud = nullptr;
	maps.clear();
	init_guess.clear();
	filtered_scan.clear();
	results.clear();
}
void ndt_mapping_base::quit() {
	try {
		input_file.close();
	} catch (std::ifstream::failure& e) {
	}
	try {
		output_file.close();
	} catch (std::ifstream::failure& e) {
	}
#ifdef EPHOS_TESTDATA_GEN
	try {
		datagen_file.close();
	} catch (std::ofstream::failure& e) {
	}
#endif
}

/**
 * Applies the transformation matrix to all point cloud elements
 * input: points to be transformed
 * output: transformed points
 * transform: transformation matrix
 */
void transformPointCloud(const PointCloud& input, PointCloud &output, Matrix4f transform)
{
	assert((input.size == output.size) && "Input and output sizes do not match");
	// TODO: make sure the sizes match before this function call
// 	if (&input != &output)
// 	{
// 		output.clear();
// 		output.resize(input.size());
// 	}
	//for (auto it = 0 ; it < input.size(); ++it)
	for (int i = 0; i < input.size; i++)
	{
		PointXYZI transformed;
		for (int row = 0; row < 3; row++)
		{
			transformed.data[row] =
				transform.data[row][0]*input.data[i].data[0] +
				transform.data[row][1]*input.data[i].data[1] +
				transform.data[row][2]*input.data[i].data[2] +
				transform.data[row][3];
		}
		output.data[i] = transformed;
	}
}

bool ndt_mapping_base::updateIntervalMT (
	double &a_l, double &f_l, double &g_l,
	double &a_u, double &f_u, double &g_u,
	double a_t, double f_t, double g_t)
{
	// Case U1 in Update Algorithm and Case a in Modified Update Algorithm [More, Thuente 1994]
	if (f_t > f_l)
	{
		a_u = a_t;
		f_u = f_t;
		g_u = g_t;
		return (false);
	}
	// Case U2 in Update Algorithm and Case b in Modified Update Algorithm [More, Thuente 1994]
	else
	if (g_t * (a_l - a_t) > 0)
	{
		a_l = a_t;
		f_l = f_t;
		g_l = g_t;
		return (false);
	}
	// Case U3 in Update Algorithm and Case c in Modified Update Algorithm [More, Thuente 1994]
	else
	if (g_t * (a_l - a_t) < 0)
	{
		a_u = a_l;
		f_u = f_l;
		g_u = g_l;

		a_l = a_t;
		f_l = f_t;
		g_l = g_t;
		return (false);
	}
	// Interval Converged
	else
		return (true);
}

double ndt_mapping_base::trialValueSelectionMT (
	double a_l, double f_l, double g_l,
	double a_u, double f_u, double g_u,
	double a_t, double f_t, double g_t)
{
	// Case 1 in Trial Value Selection [More, Thuente 1994]
	if (f_t > f_l)
	{
		// Calculate the minimizer of the cubic that interpolates f_l, f_t, g_l and g_t
		// Equation 2.4.52 [Sun, Yuan 2006]
		double z = 3 * (f_t - f_l) / (a_t - a_l) - g_t - g_l;
		double w = sqrt (z * z - g_t * g_l);
		// Equation 2.4.56 [Sun, Yuan 2006]
		double a_c = a_l + (a_t - a_l) * (w - g_l - z) / (g_t - g_l + 2 * w);

		// Calculate the minimizer of the quadratic that interpolates f_l, f_t and g_l
		// Equation 2.4.2 [Sun, Yuan 2006]
		double a_q = a_l - 0.5 * (a_l - a_t) * g_l / (g_l - (f_l - f_t) / (a_l - a_t));

		if (fabs (a_c - a_l) < fabs (a_q - a_l))
		return (a_c);
		else
		return (0.5 * (a_q + a_c));
	}
	// Case 2 in Trial Value Selection [More, Thuente 1994]
	else
	if (g_t * g_l < 0)
	{
		// Calculate the minimizer of the cubic that interpolates f_l, f_t, g_l and g_t
		// Equation 2.4.52 [Sun, Yuan 2006]
		double z = 3 * (f_t - f_l) / (a_t - a_l) - g_t - g_l;
		double w = sqrt (z * z - g_t * g_l);
		// Equation 2.4.56 [Sun, Yuan 2006]
		double a_c = a_l + (a_t - a_l) * (w - g_l - z) / (g_t - g_l + 2 * w);

		// Calculate the minimizer of the quadratic that interpolates f_l, g_l and g_t
		// Equation 2.4.5 [Sun, Yuan 2006]
		double a_s = a_l - (a_l - a_t) / (g_l - g_t) * g_l;

		if (fabs (a_c - a_t) >= fabs (a_s - a_t))
		return (a_c);
		else
		return (a_s);
	}
	// Case 3 in Trial Value Selection [More, Thuente 1994]
	else
	if (fabs (g_t) <= fabs (g_l))
	{
		// Calculate the minimizer of the cubic that interpolates f_l, f_t, g_l and g_t
		// Equation 2.4.52 [Sun, Yuan 2006]
		double z = 3 * (f_t - f_l) / (a_t - a_l) - g_t - g_l;
		double w = sqrt (z * z - g_t * g_l);
		double a_c = a_l + (a_t - a_l) * (w - g_l - z) / (g_t - g_l + 2 * w);

		// Calculate the minimizer of the quadratic that interpolates g_l and g_t
		// Equation 2.4.5 [Sun, Yuan 2006]
		double a_s = a_l - (a_l - a_t) / (g_l - g_t) * g_l;

		double a_t_next;

		if (fabs (a_c - a_t) < fabs (a_s - a_t))
		a_t_next = a_c;
		else
		a_t_next = a_s;

		if (a_t > a_l)
		return (std::min (a_t + 0.66 * (a_u - a_t), a_t_next));
		else
		return (std::max (a_t + 0.66 * (a_u - a_t), a_t_next));
	}
	// Case 4 in Trial Value Selection [More, Thuente 1994]
	else
	{
		// Calculate the minimizer of the cubic that interpolates f_u, f_t, g_u and g_t
		// Equation 2.4.52 [Sun, Yuan 2006]
		double z = 3 * (f_t - f_u) / (a_t - a_u) - g_t - g_u;
		double w = sqrt (z * z - g_t * g_u);
		// Equation 2.4.56 [Sun, Yuan 2006]
		return (a_u + (a_t - a_u) * (w - g_u - z) / (g_t - g_u + 2 * w));
	}
}

void ndt_mapping_base::buildTransformationMatrix(Matrix4f &matrix, Vec6 transform)
{
	// generating the transformation matrix componentwise with quaternions
	const float q_ha = 0.5f * transform[3];
	const float q_w = cos(q_ha);
	const float q_x = sin(q_ha);
	const float q_y = 0.0;
	const float q_z = 0.0;

	const float q_ha2 = 0.5f * transform[4];
	const float q_w2 = cos(q_ha2);
	const float q_x2 = 0.0;
	const float q_y2 = sin(q_ha2);
	const float q_z2 = 0.0;

	const float q_ha3 = 0.5f * transform[5];
	const float q_w3 = cos(q_ha3);
	const float q_x3 = 0.0;
	const float q_y3 = 0.0;
	const float q_z3 = sin(q_ha3);

	//quaternion 1 * quaternion 2
	const float r_x = q_w * q_w2 - q_x * q_x2 - q_y * q_y2 - q_z * q_z2;
	const float r_y = q_w * q_x2 + q_x * q_w2 + q_y * q_z2 - q_z * q_y2;
	const float r_z = q_w * q_y2 + q_y * q_w2 + q_z * q_x2 - q_x * q_z2;
	const float r_w = q_w * q_z2 + q_z * q_w2 + q_x * q_y2 - q_y * q_x2;

	// q1*q2 * quaternion 3
	const float r2_x = r_w * q_w3 - r_x * q_x3 - r_y * q_y3 - r_z * q_z3;
	const float r2_y = r_w * q_x3 + r_x * q_w3 + r_y * q_z3 - r_z * q_y3;
	const float r2_z = r_w * q_y3 + r_y * q_w3 + r_z * q_x3 - r_x * q_z3;
	const float r2_w = r_w * q_z3 + r_z * q_w3 + r_x * q_y3 - r_y * q_x3;

	//now compute some intermediate values for the rotationmatrix from q1*q2*q3
	const float tx  = 2.0f*r2_x;
	const float ty  = 2.0f*r2_y;
	const float tz  = 2.0f*r2_z;
	const float twx = tx*r2_w;
	const float twy = ty*r2_w;
	const float twz = tz*r2_w;
	const float txx = tx*r2_x;
	const float txy = ty*r2_x;
	const float txz = tz*r2_x;
	const float tyy = ty*r2_y;
	const float tyz = tz*r2_y;
	const float tzz = tz*r2_z;

	matrix.data[3][0] = 0.0;
	matrix.data[3][1] = 0.0;
	matrix.data[3][2] = 0.0;
	matrix.data[3][3] = 1.0;
	matrix.data[0][0] = transform[0];
	matrix.data[0][1] = transform[1];
	matrix.data[0][2] = transform[2];

	matrix.data[0][0] = 1.0f-(tyy+tzz);
	matrix.data[0][1] = txy-twz;
	matrix.data[0][2] = txz+twy;
	matrix.data[1][0] = txy+twz;
	matrix.data[1][1] = 1.0f-(txx+tzz);
	matrix.data[1][2] = tyz-twx;
	matrix.data[2][0] = txz-twy;
	matrix.data[2][1] = tyz+twx;
	matrix.data[2][2] = 1.0f-(txx+tyy);
}

// from /usr/include/pcl-1.7/pcl/registration/impl/ndt.hpp
double ndt_mapping_base::computeStepLengthMT (
	const Vec6 &x, Vec6 &step_dir, double step_init, double step_max,
	double step_min, double &score, Vec6 &score_gradient, Mat66 &hessian,
	PointCloud& trans_cloud)
{
	// Set the value of phi(0), Equation 1.3 [More, Thuente 1994]
	double phi_0 = -score;
	// Set the value of phi'(0), Equation 1.3 [More, Thuente 1994]
	double d_phi_0 = -(dot_product6(score_gradient, step_dir));
	Vec6  x_t = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
	if (d_phi_0 >= 0)
	{
		// Not a decent direction
		if (d_phi_0 == 0)
			return 0;
		else
		{
			// Reverse step direction and calculate optimal step.
			d_phi_0 *= -1;
			for (int i = 0; i < 6; i++)
			step_dir[i] *= -1;

		}
	}
	// The Search Algorithm for T(mu) [More, Thuente 1994]
	int max_step_iterations = 10;
	int step_iterations = 0;
	// Sufficient decreace constant, Equation 1.1 [More, Thuete 1994]
	double mu = 1.e-4;
	// Curvature condition constant, Equation 1.2 [More, Thuete 1994]
	double nu = 0.9;
	// Initial endpoints of Interval I,
	double a_l = 0, a_u = 0;
	// Auxiliary function psi is used until I is determined ot be a closed interval, Equation 2.1 [More, Thuente 1994]
	double f_l = auxilaryFunction_PsiMT (a_l, phi_0, phi_0, d_phi_0, mu);
	double g_l = auxilaryFunction_dPsiMT (d_phi_0, d_phi_0, mu);

	double f_u = auxilaryFunction_PsiMT (a_u, phi_0, phi_0, d_phi_0, mu);
	double g_u = auxilaryFunction_dPsiMT (d_phi_0, d_phi_0, mu);
	// Check used to allow More-Thuente step length calculation to be skipped by making step_min == step_max
	bool interval_converged = (step_max - step_min) > 0, open_interval = true;
	double a_t = step_init;
	a_t = std::min (a_t, step_max);
	a_t = std::max (a_t, step_min);
	for (int i = 0; i < 6; i++)
		x_t[i] = x[i] + step_dir[i] * a_t;

	buildTransformationMatrix(final_transformation_, x_t);
	intermediate_transformations_.push_back(final_transformation_);
	// New transformed point cloud
	transformPointCloud(*input_cloud, trans_cloud, final_transformation_);
	// Updates score, gradient and hessian.  Hessian calculation is unessisary but testing showed that most step calculations use the
	// initial step suggestion and recalculation the reusable portions of the hessian would intail more computation time.
	score = computeDerivatives (score_gradient, hessian, trans_cloud, x_t, true);
	// Calculate phi(alpha_t)
	double phi_t = -score;
	// Calculate phi'(alpha_t)
	double d_phi_t = -(dot_product6(score_gradient, step_dir));
	// Calculate psi(alpha_t)
	double psi_t = auxilaryFunction_PsiMT (a_t, phi_t, phi_0, d_phi_0, mu);
	// Calculate psi'(alpha_t)
	double d_psi_t = auxilaryFunction_dPsiMT (d_phi_t, d_phi_0, mu);
	// Iterate until max number of iterations, interval convergance or a value satisfies the sufficient decrease, Equation 1.1, and curvature condition, Equation 1.2 [More, Thuente 1994]
	while (!interval_converged && step_iterations < max_step_iterations && !(psi_t <= 0 /*Sufficient Decrease*/ && d_phi_t <= -nu * d_phi_0 /*Curvature Condition*/))
	{
		// Use auxilary function if interval I is not closed
		if (open_interval)
		{
			a_t = trialValueSelectionMT (
				a_l, f_l, g_l,
				a_u, f_u, g_u,
				a_t, psi_t, d_psi_t);
		}
		else
		{
			a_t = trialValueSelectionMT (
				a_l, f_l, g_l,
				a_u, f_u, g_u,
				a_t, phi_t, d_phi_t);
		}
		a_t = std::min (a_t, step_max);
		a_t = std::max (a_t, step_min);
		for (int row = 0; row < 6; row++)
			x_t[row] = x[row] + step_dir[row] * a_t;

		buildTransformationMatrix(final_transformation_, x_t);
		intermediate_transformations_.push_back(final_transformation_);
		// New transformed point cloud
		// Done on final cloud to prevent wasted computation
		transformPointCloud (*input_cloud, trans_cloud, final_transformation_);
		// Updates score, gradient. Values stored to prevent wasted computation.
		score = computeDerivatives (score_gradient, hessian, trans_cloud, x_t, false);
		// Calculate phi(alpha_t+)
		phi_t = -score;
		// Calculate phi'(alpha_t+)
		d_phi_t = -(dot_product6(score_gradient, step_dir));
		// Calculate psi(alpha_t+)
		psi_t = auxilaryFunction_PsiMT (a_t, phi_t, phi_0, d_phi_0, mu);
		// Calculate psi'(alpha_t+)
		d_psi_t = auxilaryFunction_dPsiMT (d_phi_t, d_phi_0, mu);
		// Check if I is now a closed interval
		if (open_interval && (psi_t <= 0 && d_psi_t >= 0))
		{
			open_interval = false;
			// Converts f_l and g_l from psi to phi
			f_l = f_l + phi_0 - mu * d_phi_0 * a_l;
			g_l = g_l + mu * d_phi_0;
			// Converts f_u and g_u from psi to phi
			f_u = f_u + phi_0 - mu * d_phi_0 * a_u;
			g_u = g_u + mu * d_phi_0;
		}

		if (open_interval)
		{
			// Update interval end points using Updating Algorithm [More, Thuente 1994]
			interval_converged = updateIntervalMT (
				a_l, f_l, g_l,
				a_u, f_u, g_u,
				a_t, psi_t, d_psi_t);
		}
		else
		{
			// Update interval end points using Modified Updating Algorithm [More, Thuente 1994]
			interval_converged = updateIntervalMT (
				a_l, f_l, g_l,
				a_u, f_u, g_u,
				a_t, phi_t, d_phi_t);
		}
		step_iterations++;
	}
	// gradients are required for step length determination
	// so derivative and transform data is stored for the next iteration.
	if (step_iterations)
		computeHessian (hessian, trans_cloud, x_t);

	return a_t;
}

void ndt_mapping_base::eulerAngles(Matrix4f trans, Vec3 &result)
{
	Vec3 res;
	const int i = 0;
	const int j = 1;
	const int k = 2;
	res[0] = atan2(trans.data[j][k], trans.data[k][k]);
	double n1 = trans.data[i][i];
	double n2 = trans.data[i][j];
	double c2 = sqrt(n1*n1+n2*n2);
	if(res[0]>0.0) {
		if(res[0] > 0.0) {
			res[0] -= PI;
		}
		else {
			res[0] += PI;
		}
		res[1] = atan2(-trans.data[i][k], -c2);
	}
	else
		res[1] = atan2(-trans.data[i][k], c2);
	double s1 = sin(res[0]);
	double c1 = cos(res[0]);
	res[2] = atan2(s1*trans.data[k][i]-c1*trans.data[j][i], c1*trans.data[j][j] - s1 * trans.data[k][j]);
	result[0] = -res[0];
	result[1] = -res[1];
	result[2] = -res[2];
}

void ndt_mapping_base::computeTransformation(PointCloud &output, const Matrix4f &guess)
{
	nr_iterations_ = 0;
	converged_ = false;
	double gauss_c1, gauss_c2, gauss_d3;
	// Initializes the guassian fitting parameters (eq. 6.8) [Magnusson 2009]
	gauss_c1 = 10 * (1 - outlier_ratio_);
	gauss_c2 = outlier_ratio_ / pow (resolution_, 3);
	gauss_d3 = -log (gauss_c2);
	gauss_d1_ = -log ( gauss_c1 + gauss_c2 ) - gauss_d3;
	gauss_d2_ = -2 * log ((-log ( gauss_c1 * exp ( -0.5 ) + gauss_c2 ) - gauss_d3) / gauss_d1_);
	// Initialise final transformation to the guessed one
	final_transformation_ = guess;
	// Apply guessed transformation prior to search for neighbours
	transformPointCloud (output, output, guess);
	// Initialize Point Gradient and Hessian
	memset(point_gradient_.data, 0, sizeof(double) * 3 * 6);
	point_gradient_.data[0][0] = 1.0;
	point_gradient_.data[1][1] = 1.0;
	point_gradient_.data[2][2] = 1.0;
	memset(point_hessian_.data, 0, sizeof(double) * 18 * 6);
	// Convert initial guess matrix to 6 element transformation vector
	Vec6 p, delta_p, score_gradient;
	p[0] = final_transformation_.data[0][4];
	p[1] = final_transformation_.data[1][4];
	p[2] = final_transformation_.data[2][4];
	Vec3 ea;
	eulerAngles(final_transformation_, ea);
	p[3] = ea[0];
	p[4] = ea[1];
	p[5] = ea[2];
	Mat66 hessian;
	double score = 0;
	double delta_p_norm;
	// Calculate derivates of initial transform vector, subsequent derivative calculations are done in the step length determination.
	score = computeDerivatives (score_gradient, hessian, output, p);
	while (!converged_)
	{
		// Store previous transformation
		previous_transformation_ = transformation_;
		// Negative for maximization as opposed to minimization
		Vec6 neg_grad = {
			-score_gradient[0],
			-score_gradient[1],
			-score_gradient[2],
			-score_gradient[3],
			-score_gradient[4],
			-score_gradient[5]
		};
		solve (delta_p, hessian, neg_grad);
		//Calculate step length with guarnteed sufficient decrease [More, Thuente 1994]
		delta_p_norm = sqrt(delta_p[0] * delta_p[0] +
				delta_p[1] * delta_p[1] +
				delta_p[2] * delta_p[2] +
				delta_p[3] * delta_p[3] +
				delta_p[4] * delta_p[4] +
				delta_p[5] * delta_p[5]);
		delta_p_norm = 1;
		if (delta_p_norm == 0 || delta_p_norm != delta_p_norm)
		{
			transformation_probability_ = score / static_cast<double> (input_cloud->size);
			converged_ = delta_p_norm == delta_p_norm;
			return;
		}

		delta_p[0] /= delta_p_norm;
		delta_p[1] /= delta_p_norm;
		delta_p[2] /= delta_p_norm;
		delta_p[3] /= delta_p_norm;
		delta_p[4] /= delta_p_norm;
		delta_p[5] /= delta_p_norm;

		delta_p_norm = computeStepLengthMT (p, delta_p, delta_p_norm, step_size_, transformation_epsilon_ / 2, score, score_gradient, hessian, output);
		delta_p[0] *= delta_p_norm;
		delta_p[1] *= delta_p_norm;
		delta_p[2] *= delta_p_norm;
		delta_p[3] *= delta_p_norm;
		delta_p[4] *= delta_p_norm;
		delta_p[5] *= delta_p_norm;

		buildTransformationMatrix(transformation_, delta_p);
		intermediate_transformations_.push_back(transformation_);
		p[0] = p[0] + delta_p[0];
		p[1] = p[1] + delta_p[1];
		p[2] = p[2] + delta_p[2];
		p[3] = p[3] + delta_p[3];
		p[4] = p[4] + delta_p[4];
		p[5] = p[5] + delta_p[5];
		if (nr_iterations_ > max_iterations_ ||
			(nr_iterations_ && (fabs (delta_p_norm) < transformation_epsilon_)))
		{
			converged_ = true;
		}
		nr_iterations_++;
	}

	// Store transformation probability.  The realtive differences within each scan registration are accurate
	// but the normalization constants need to be modified for it to be globally accurate
	transformation_probability_ = score / static_cast<double>(input_cloud->size);
}



void ndt_mapping_base::ndt_align(const Matrix4f& guess)
{

	initCompute();
	// Copy the point data to output
	//PointCloud output(*input_cloud);
	// TODO check correctness
	PointCloud output = {
		new PointXYZI[input_cloud->capacity],
		input_cloud->size,
		input_cloud->capacity
	};
	std::memcpy(output.data, input_cloud->data, sizeof(PointXYZI)*input_cloud->size);
	// Perform the actual transformation computation
	converged_ = false;
	final_transformation_ = transformation_ = previous_transformation_ = {
		{{ 1.0, 0.0, 0.0, 0.0 },
		 { 0.0, 1.0, 0.0, 0.0 },
		 { 0.0, 0.0, 1.0, 0.0 },
		 { 0.0, 0.0, 0.0, 1.0 }}
	};
	// Right before we estimate the transformation, we set all the point.data[3] values to 1
	// to aid the rigid transformation
	for (int i = 0; i < output.size; i++) {
		output.data[i].data[3] = 1.0;
	}
	computeTransformation(output, guess);
	// free allocated memory
	cleanupCompute();
	delete[] output.data;
}


CallbackResult ndt_mapping_base::partial_points_callback(
	PointCloud& input_cloud,
	Matrix4f& init_guess,
	PointCloud& target_cloud)
{
	CallbackResult result;
	this->input_cloud = &input_cloud;
	this->target_cloud = &target_cloud;
	intermediate_transformations_.clear();
	ndt_align(init_guess);
	result.final_transformation = final_transformation_;
	result.intermediate_transformations = intermediate_transformations_;
	result.converged = converged_;
	return result;
}

int ndt_mapping_base::read_next_testcases(int count)
{
	int i;
	// free memory used in the previous test case and allocate new one
	maps.resize(count);
	filtered_scan.resize(count);
	init_guess.resize(count);
	results.resize(count);
	// parse the test cases
	for (i = 0; (i < count) && (read_testcases < testcases); i++,read_testcases++)
	{
		try {
			parseInitGuess(input_file, init_guess[i]);
			parseFilteredScan(input_file, filtered_scan[i]);
			parseFilteredScan(input_file, maps[i]);
		} catch (std::ios_base::failure& e) {
			std::cerr << e.what() << std::endl;
			exit(-3);
		}
	}
	return i;
}
void ndt_mapping_base::cleanupTestcases(int count) {
	// free memory allocated by parsers
	for (int i = 0; i < count; i++) {
		delete[] filtered_scan[i].data;
	}
	filtered_scan.resize(0);
	for (int i = 0; i < count; i++) {
		delete[] maps[i].data;
	}
	maps.resize(0);
	init_guess.resize(0);
	results.resize(0);
}
void  ndt_mapping_base::parseFilteredScan(std::ifstream& input_file, PointCloud& pointcloud) {
	int32_t cloudSize;
	try {
		input_file.read((char*)&cloudSize, sizeof(int32_t));
		// TODO make sure to not create a memory leak here
		//pointcloud.clear();
		pointcloud.data = new PointXYZI[cloudSize];
		pointcloud.size = cloudSize;
		pointcloud.capacity = cloudSize;
		for (int i = 0; i < cloudSize; i++)
		{
			PointXYZI p;
			input_file.read((char*)&p.data[0], sizeof(float));
			input_file.read((char*)&p.data[1], sizeof(float));
			input_file.read((char*)&p.data[2], sizeof(float));
			input_file.read((char*)&p.data[3], sizeof(float));
			pointcloud.data[i] = p;
		}
	}  catch (std::ifstream::failure& e) {
		throw std::ios_base::failure("Error reading filtered scan");
	}
}


void ndt_mapping_base::parseInitGuess(std::ifstream& input_file, Matrix4f& initGuess) {
	try {
		for (int h = 0; h < 4; h++)
			for (int w = 0; w < 4; w++)
				input_file.read((char*)&(initGuess.data[h][w]),sizeof(float));
	}  catch (std::ifstream::failure& e) {
		throw std::ios_base::failure("Error reading initial guess");
	}
}

/**
 * Reads the next reference matrix.
 */
void ndt_mapping_base::parseResult(std::ifstream& output_file, CallbackResult& result) {
	try {
		for (int h = 0; h < 4; h++)
			for (int w = 0; w < 4; w++)
			{
				float m;
				output_file.read((char*)&m, sizeof(float));
				result.final_transformation.data[h][w] = m;
			}
		double fitness;
		output_file.read((char*)&fitness, sizeof(double));
		result.fitness_score = fitness;
		bool converged;
		output_file.read((char*)&converged, sizeof(bool));
		result.converged = converged;
	}  catch (std::ifstream::failure& e) {
		throw std::ios_base::failure("Error reading result.");
	}
}

void ndt_mapping_base::parseIntermediateResults(std::ifstream& output_file, CallbackResult& result) {

	try {
		int resultNo;
		output_file.read((char*)&resultNo, sizeof(int32_t));
		result.intermediate_transformations.resize(resultNo);
		for (int i = 0; i < resultNo; i++) {
			Matrix4f& m = result.intermediate_transformations[i];
			for (int h = 0; h < 4; h++) {
				for (int w = 0; w < 4; w++) {
					output_file.read((char*)&(m.data[h][w]), sizeof(float));
				}
			}
		}
	} catch (std::ifstream::failure& e) {
		throw std::ios_base::failure("Error reading voxel grid.");
	}
}
void ndt_mapping_base::writeResult(std::ofstream& output_file, CallbackResult& result) {
	try {
		Matrix4f& m = result.final_transformation;
		for (int h = 0; h < 4; h++) {
			for (int w = 0; w < 4; w++) {
				output_file.write((char*)&(m.data[h][w]), sizeof(float));
			}
		}
		double fitness = result.fitness_score;
		output_file.write((char*)&fitness, sizeof(double));
		bool converged = result.converged;
		output_file.write((char*)&converged, sizeof(bool));
	} catch (std::ofstream::failure& e) {
		throw std::ios_base::failure("Error writeing result.");
	}
}
void ndt_mapping_base::writeIntermediateResults(std::ofstream& output_file, CallbackResult& result) {

	try {
		int resultNo = result.intermediate_transformations.size();
		output_file.write((char*)&resultNo, sizeof(int32_t));
		for (int i = 0; i < resultNo; i++) {
			Matrix4f& m = result.intermediate_transformations[i];
			for (int h = 0; h < 4; h++) {
				for (int w = 0; w < 4; w++) {
					output_file.write((char*)&(m.data[h][w]), sizeof(float));
				}
			}
		}
	} catch (std::ofstream::failure& e) {
		throw std::ios_base::failure("Error writing voxel grid. ");
	}
}
#ifdef EPHOS_TESTDATA_LEGACY
int ndt_mapping_base::read_testdata_signature(std::ifstream& input_file, std::ifstream& output_file)
{
	int32_t number;
	try {
		input_file.read((char*)&number, sizeof(int32_t));
	} catch (std::ifstream::failure) {
		throw std::ios_base::failure("Error reading number of test cases");
	}
	return number;
}
#else // EPHOS_TESTDATA_LEGACY
int ndt_mapping_base::read_testdata_signature(std::ifstream& input_file, std::ifstream& output_file)
{
	int32_t number1, number2, zero, version1, version2;
	try {
		input_file.read((char*)&zero, sizeof(int32_t));
		input_file.read((char*)&version1, sizeof(int32_t));
		input_file.read((char*)&number1, sizeof(int32_t));
	} catch (std::ifstream::failure&) {
		throw std::ios_base::failure("Error reading the input data signature");
	}
	if (zero != 0x0) {
		throw std::ios_base::failure(
			"Misformatted input test data signature. You may be using legacy test data");
	}
	if (version1 != 0x1) {
		throw std::ios_base::failure(
			std::string(
				"Misformatted input test data signature. "
				"Expected test data version 1. "
				"Instead got version ") + std::to_string(version1));
	}
	if (number1 < 0 || number1 > 10000) {
		throw std::ios_base::failure(
			std::string("Unreasonable number of test cases (") +
			std::to_string(number1) +
			std::string(") in input test data"));
	}
	try {
		output_file.read((char*)&zero, sizeof(int32_t));
		output_file.read((char*)&version2, sizeof(int32_t));
		output_file.read((char*)&number2, sizeof(int32_t));
	} catch (std::ifstream::failure) {
		throw std::ios_base::failure("Error reading the output test data signature");
	}
	if (zero != 0x0) {
		throw std::ios_base::failure(
			"Misformatted output test data signature. You may be using legacy test data");
	}
	if (version2 != 0x1) {
		throw std::ios_base::failure(
			std::string(
				"Misformatted output test data signature. "
				"Expected test data version 1. "
				"Instead got version ") +
			std::to_string(version2));
	}
	if (number2 != number1) {
		throw std::ios_base::failure(
			std::string("Number of test cases in output test data (") +
			std::to_string(number2) +
			std::string(") does not match number of test cases input test data (") +
			std::to_string(number1) + std::string(")"));
	}
	return number1;
}
#endif // !EPHOS_TESTDATA_LEGACY

void ndt_mapping_base::run(int p) {
	std::cout << "executing for " << testcases << " test cases" << std::endl;
	start_timer();
	pause_timer();
	while (read_testcases < testcases)
	{
		int count = read_next_testcases(p);

		resume_timer();
		for (int i = 0; i < count; i++)
		{
			// actual kernel invocation
			results[i] = partial_points_callback(
				filtered_scan[i],
				init_guess[i],
				maps[i]
			);
		}
		pause_timer();
		check_next_outputs(count);
		cleanupTestcases(count);
	}
	stop_timer();
}

void ndt_mapping_base::check_next_outputs(int count)
{
	CallbackResult reference;
	for (int i = 0; i < count; i++)
	{
		try {
			parseResult(output_file, reference);
#ifndef EPHOS_TESTDATA_LEGACY
			parseIntermediateResults(output_file, reference);
#endif
#ifdef EPHOS_TESTDATA_GEN
			writeResult(datagen_file, results[i]);
			std::cout << "inter: " << results[i].intermediate_transformations.size() << std::endl;
			writeIntermediateResults(datagen_file, results[i]);
#endif
		} catch (std::ios_base::failure& e) {
			std::cerr << e.what() << std::endl;
			exit(-3);
		}
		if (results[i].converged != reference.converged)
		{
			error_so_far = true;
		}
		// compare the matrices
		for (int h = 0; h < 4; h++) {
			// test for nan
			for (int w = 0; w < 4; w++) {
				if (std::isnan(results[i].final_transformation.data[h][w]) !=
					std::isnan(reference.final_transformation.data[h][w])) {
					error_so_far = true;
				}
			}
			// compare translation
			float delta = std::fabs(results[i].final_transformation.data[h][3] -
				reference.final_transformation.data[h][3]);
			if (delta > max_delta) {
				max_delta = delta;
				if (delta > EPHOS_MAX_TRANSLATION_EPS) {
					error_so_far = true;
				}
			}
		}
		// compare transformed points
		PointXYZI origin = {
			{ 0.724f, 0.447f, 0.525f, 1.0f }
		};
		PointXYZI resPoint = {
			{ 0.0f, 0.0f, 0.0f, 0.0f }
		};
		PointXYZI refPoint = {
			{ 0.0f, 0.0f, 0.0f, 0.0f }
		};
		for (int h = 0; h < 4; h++) {
			for (int w = 0; w < 4; w++) {
				resPoint.data[h] += results[i].final_transformation.data[h][w]*origin.data[w];
				refPoint.data[h] += reference.final_transformation.data[h][w]*origin.data[w];
			}
		}
		for (int w = 0; w < 4; w++) {
			float delta = std::fabs(resPoint.data[w] - refPoint.data[w]);
			if (delta > max_delta) {
				max_delta = delta;
				if (delta > EPHOS_MAX_EPS) {
					error_so_far = true;
				}
			}
		}
		std::ostringstream sError;
		int caseErrorNo = 0;
		for (int w = 0; w < 4; w++) {
			float delta = std::fabs(resPoint.data[w] - refPoint.data[w]);
			if (delta > max_delta) {
				max_delta = delta;
				if (delta > EPHOS_MAX_ROTATION_EPS) {
					error_so_far = true;
				}
			}
			if (delta > EPHOS_MAX_ROTATION_EPS) {
				sError << " mismatch vector[" << w << "]: ";
				sError << refPoint.data[w] << " should be " << resPoint.data[w] << std::endl;
				caseErrorNo += 1;
			}
		}
		if (caseErrorNo > 0) {
			std::cout << "Errors for test case " << read_testcases - count + i;
			std::cout << sError.str() << std::endl;
		}
	}
}

bool ndt_mapping_base::check_output() {
	std::cout << "checking output \n";
	// check for error
	std::cout << "max delta: " << max_delta << "\n";
	return !error_so_far;
}
