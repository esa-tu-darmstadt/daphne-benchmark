/**
 * Author:  Florian Stock, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2018
 * Author:  Thilo Gabel, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2019
 * License: Apache 2.0 (see attachached File)
 */
#ifndef EPHOS_KERNEL_H
#define EPHOS_KERNEL_H

#include "benchmark.h"
#include "datatypes.h"
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <limits>
#include <cstring>
#include <chrono>

// maximum allowed deviation from reference
#define MAX_TRANSLATION_EPS 0.001
//#define MAX_ROTATION_EPS 0.001
#define MAX_ROTATION_EPS 0.01

//#define MAX_EPS 2

#define PI 3.1415926535897932384626433832795

class ndt_mapping : public kernel {
private:
	// the number of testcases read
	int read_testcases = 0;
	PointCloud* filtered_scan_ptr = nullptr;
	Matrix4f* init_guess = nullptr;
	CallbackResult* results = nullptr;
	PointCloud* maps = nullptr;
	// testcase and result stream
	std::ifstream input_file, output_file;
	// whether an abnormal deviation has been detected
	bool error_so_far = false;
	// maximum deviation from the reference data so far
	double max_delta = 0.0;
	// ndt parameters
	double outlier_ratio_ = 0.55;
	float resolution_ = 1.0;
	double trans_eps = 0.01; //Transformation epsilon
	double step_size_ = 0.1; // Step size
	int iter = 30;  // Maximum iterations
	Matrix4f final_transformation_, transformation_, previous_transformation_;
	bool converged_ = false;
	int nr_iterations_ = 0;
	Vec3 h_ang_a2_, h_ang_a3_,
	h_ang_b2_, h_ang_b3_,
	h_ang_c2_, h_ang_c3_,
	h_ang_d1_, h_ang_d2_, h_ang_d3_,
	h_ang_e1_, h_ang_e2_, h_ang_e3_,
	h_ang_f1_, h_ang_f2_, h_ang_f3_;
	Vec3 j_ang_a_, j_ang_b_, j_ang_c_, j_ang_d_, j_ang_e_, j_ang_f_, j_ang_g_, j_ang_h_;
	Mat36 point_gradient_;
	Mat186 point_hessian_;
	double gauss_d1_, gauss_d2_;
	double trans_probability_;
	double transformation_epsilon_ = 0.1;
	int max_iterations_ = 0;
	// point clouds
	PointCloud* input_ = nullptr;
	PointCloud* target_ = nullptr;
	// voxel grid spanning over all points
	VoxelGrid target_cells_;
	// voxel grid extend
	PointXYZI minVoxel, maxVoxel;
	int voxelDimension[3];
public:
	virtual void init();
	virtual void run(int p = 1);
	virtual bool check_output();
protected:
	/**
	 * Reads the number of testcases in the data file
	 */
	int read_number_testcases(std::ifstream& input_file);
	/**
	 * Reads the next testcases.
	 * count: number of datasets to read
	 * return: number of data sets actually read.
	 */
	virtual int read_next_testcases(int count);
	/**
	 * Reads and compares algorithm results with the respective reference.
	 * count: number of testcase results to compare
	 */
	virtual void check_next_outputs(int count);
	/**
	 * Reduces a multi dimensional voxel grid index to one dimension.
	 */
	inline int linearizeAddr(const int x, const int y, const int z);
	/**
	 * Reduces a coordinate to a voxel grid index.
	 */
	inline int linearizeCoord(const float x, const float y, const float z);

	double updateDerivatives (Vec6 &score_gradient,
		Mat66 &hessian,
		Vec3 &x_trans, Mat33 &c_inv,
		bool compute_hessian = true);
	void computePointDerivatives (Vec3 &x, bool compute_hessian = true);
	void computeHessian (Mat66 &hessian,
		PointCloudSource &trans_cloud, Vec6 &);
	void updateHessian (Mat66 &hessian, Vec3 &x_trans, Mat33 &c_inv);
	double computeDerivatives (Vec6 &score_gradient,
		Mat66 &hessian,
		PointCloudSource &trans_cloud,
		Vec6 &p,
		bool compute_hessian = true );
	bool updateIntervalMT (double &a_l, double &f_l, double &g_l,
		double &a_u, double &f_u, double &g_u,
		double a_t, double f_t, double g_t);
	double trialValueSelectionMT (double a_l, double f_l, double g_l,
		double a_u, double f_u, double g_u,
		double a_t, double f_t, double g_t);
	double computeStepLengthMT (const Vec6 &x, Vec6 &step_dir, double step_init, double step_max,
				double step_min, double &score, Vec6 &score_gradient, Mat66 &hessian,
				PointCloudSource &trans_cloud);
	void computeTransformation(PointCloud &output, const Matrix4f &guess);
	void computeAngleDerivatives (Vec6 &p, bool compute_hessian = true);
	void ndt_align (const Matrix4f& guess);
	/**
	 * Performs point cloud specific voxel grid initialization.
	 */
	void initCompute();
	void buildTransformationMatrix(Matrix4f &matrix, Vec6 transform);
	/**
	 * Computes the eulerangles from an rotation matrix.
	 */
	void eulerAngles(Matrix4f transform, Vec3 &result);
	CallbackResult partial_points_callback(PointCloud &input_cloud, Matrix4f &init_guess, PointCloud& target_cloud);
	/**
	 * Helper function to select near voxels.
	 */
	int voxelRadiusSearch(
		VoxelGrid &grid, const PointXYZI& point, double radius,
		std::vector<Voxel*>& indices);
};
#endif // EPHOS_KERNEL_H