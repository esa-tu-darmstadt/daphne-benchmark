/**
 * Author:  Florian Stock, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2018
 * Author:  Thilo Gabel, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2019
 * License: Apache 2.0 (see attached files)
 */
#ifndef EPHOS_NDT_MAPPING_BASE_H
#define EPHOS_NDT_MAPPING_BASE_H


#include <cmath>
#include <iostream>
#include <sstream>
#include <fstream>
#include <limits>
#include <cstring>
#include <chrono>
#include <stdexcept>
#include <vector>

#include "datatypes_base.h"
#include "common/benchmark.h"

// maximum allowed deviation from reference
#define EPHOS_MAX_TRANSLATION_EPS 0.001
#define EPHOS_MAX_ROTATION_EPS 1.8
#define EPHOS_MAX_EPS 2

class ndt_mapping_base : public benchmark {
protected:
	// the number of testcases read
	int read_testcases;
	std::ifstream input_file, output_file;
	std::ofstream datagen_file;
	// indicates a discrete deviation
	bool error_so_far;
	// continuous deviation from the reference
	double max_delta;
	// ndt parameters
	double outlier_ratio_ = 0.55;
	float  resolution_    = 1.0;
	double trans_eps_      = 0.01; //Transformation epsilon
	double step_size_     = 0.1;  // Step size

	int iter = 30, max_iterations_; // Maximum iterations
	Matrix4f final_transformation_, transformation_, previous_transformation_;
	std::vector<Matrix4f> intermediate_transformations_;
	bool converged_;
	int nr_iterations_;
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
	double transformation_probability_;
	double transformation_epsilon_ = 0.1;

	std::vector<PointCloud> filtered_scan;
	std::vector<PointCloud> maps;
	// starting transformation matrix
	std::vector<Matrix4f> init_guess;
	// algorithm results
	std::vector<CallbackResult> results;
	// intermediate results
	std::vector<VoxelGrid> grids;
	// point clouds
	PointCloud* input_cloud = nullptr;
	PointCloud* target_cloud = nullptr;
	VoxelGrid target_grid;
	//PointCloud* filtered_scan_ptr = nullptr;

	// voxel grid extends
	//PointXYZI minVoxel, maxVoxel;
	//int voxelDimension[3];
public:
	ndt_mapping_base();
	virtual ~ndt_mapping_base();
public:
	virtual void init();
	virtual void quit();
	virtual void run(int p = 1);
	virtual bool check_output();
protected:
	/**
	 * Reads the next reference matrix.
	 */
	virtual void parseResult(std::ifstream& output_file, CallbackResult& result);
	/**
	 * Reads the next voxel grid.
	 */
	virtual void parseIntermediateResults(std::ifstream& output_file, CallbackResult& result);
	/**
	 * Writes the next reference matrix.
	 */
	virtual void writeResult(std::ofstream& output_file, CallbackResult& result);
	/**
	 * Writes the next voxel grid.
	 */
	virtual void writeIntermediateResults(std::ofstream& output_file, CallbackResult& result);
	/**
	 * Reads the next point cloud.
	 */
	virtual void  parseFilteredScan(std::ifstream& input_file, PointCloud& pointcloud);
	/**
	 * Reads the next initilization matrix.
	 */
	virtual void parseInitGuess(std::ifstream& input_file, Matrix4f& initGuess);
	/**
	 * Reads the number of testcases in the data file
	 */
	virtual int read_number_testcases(std::ifstream& input_file);
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
	 * Frees resources used during processing of a batch of test cases.
	 * count: number of cases in the test batch
	 */
	virtual void cleanupTestcases(int count);
	/**
	 * Reduces a multi dimensional voxel grid index to one dimension.
	 */
	int linearizeAddr(const int x, const int y, const int z);
	/**
	 * Reduces a coordinate to a voxel grid index.
	 */
	int linearizeCoord(const float x, const float y, const float z);
	/**
	 * Helper function to calculate the dot product of two vectors.
	 */
	double dot_product(Vec3 &a, Vec3 &b);
	/**
	 * Helper function to calculate the dot product of two vectors.
	 */
	double dot_product6(Vec6 &a, Vec6 &b);

// 	int voxelRadiusSearch(VoxelGrid& grid, const PointXYZI& point,
// 		double radius, std::vector<Voxel*>& indices);
//
// 	double updateDerivatives (Vec6 &score_gradient,
// 						Mat66 &hessian,
// 						Vec3 &x_trans, Mat33 &c_inv,
// 						bool compute_hessian = true);
//
// 	void computePointDerivatives (Vec3 &x, bool compute_hessian = true);
	virtual void computeHessian (Mat66 &hessian, PointCloud& trans_cloud, Vec6&) = 0;
// 	void updateHessian (Mat66 &hessian, Vec3 &x_trans, Mat33 &c_inv);
//
	virtual double computeDerivatives(Vec6& score_gradient,
		Mat66& hessian,
		PointCloud& trans_cloud,
		Vec6& p,
		bool compute_hessian = true) = 0;

	/**
	 * Initializes the transformation computation for a test case.
	 */
	virtual void initCompute() = 0;
	/**
	 * Frees resources used during the transformation computation of a test case.
	 */
	virtual void cleanupCompute() = 0;

	virtual bool updateIntervalMT (double &a_l, double &f_l, double &g_l,
		double &a_u, double &f_u, double &g_u,
		double a_t, double f_t, double g_t);

	virtual double trialValueSelectionMT (double a_l, double f_l, double g_l,
		double a_u, double f_u, double g_u,
		double a_t, double f_t, double g_t);

	virtual double computeStepLengthMT (const Vec6 &x, Vec6 &step_dir, double step_init, double step_max,
		double step_min, double &score, Vec6 &score_gradient, Mat66 &hessian,
		PointCloud& trans_cloud);
	/**
	 * Entry point for actually computing the transformation matrix
	 * after setup of all input, output and intermediate structures.
	 * output: output point cloud
	 * guess: transformation to start with
	 */
	virtual void computeTransformation(PointCloud &output, const Matrix4f &guess);
	/**
	 * Second entry point for a test case.
	 * Sets up structures and calls computeTransformation().
	 * guess: transformation to start with
	 */
	virtual void ndt_align(const Matrix4f& guess);

	virtual void buildTransformationMatrix(Matrix4f &matrix, Vec6 transform);
	/**
	 * Computes the eulerangles from an rotation matrix.
	 */
	virtual void eulerAngles(Matrix4f transform, Vec3 &result);
	/**
	 * Entry point for a test case.
	 * Sets up pointers and calls ndt_align()
	 */
	virtual CallbackResult partial_points_callback(
		PointCloud &input_cloud,
		Matrix4f &init_guess,
		PointCloud& target_cloud
	);
};


#endif // EPHOS_NDT_MAPPING_BASE_H

