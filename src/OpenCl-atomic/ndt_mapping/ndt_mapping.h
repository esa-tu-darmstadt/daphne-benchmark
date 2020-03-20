/**
 * Author:  Florian Stock, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2018
 * Author:  Leonardo Solis, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2018
 * Author:  Thilo Gabel, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2019
 * License: Apache 2.0 (see attachached File)
 */
#ifndef EPHOS_NDT_MAPPING_H
#define EPHOS_NDT_MAPPING_H


#include <cmath>
#include <iostream>
#include <sstream>
#include <fstream>
#include <limits>
#include <cstring>
#include <chrono>
#include <stdexcept>

#include "datatypes.h"

#include "common/benchmark.h"
#include "common/compute_tools.h"


#define STRINGIZE2(s) #s
#define STRINGIZE(s) STRINGIZE2(s)
#define EPHOS_KERNEL_WORK_GROUP_SIZE_S STRINGIZE(EPHOS_KERNEL_WORK_GROUP_SIZE)

// maximum allowed deviation from reference
#define MAX_TRANSLATION_EPS 0.001
#define MAX_ROTATION_EPS 1.8
#define MAX_EPS 2

// opencl platform hints
#if defined(EPHOS_PLATFORM_HINT)
#define EPHOS_PLATFORM_HINT_S STRINGIZE(EPHOS_PLATFORM_HINT)
#else
#define EPHOS_PLATFORM_HINT_S ""
#endif

#if defined(EPHOS_DEVICE_HINT)
#define EPHOS_DEVICE_HINT_S STRINGIZE(EPHOS_DEVICE_HINT)
#else
#define EPHOS_DEVICE_HINT_S ""
#endif

#if defined(EPHOS_DEVICE_TYPE)
#define EPHOS_DEVICE_TYPE_S STRINGIZE(EPHOS_DEVICE_TYPE)
#else
#define EPHOS_DEVICE_TYPE_S ""
#endif



class ndt_mapping : public benchmark {
private:
	// the number of testcases read
	int read_testcases = 0;
	std::ifstream input_file, output_file;
	// indicates a discrete deviation
	bool error_so_far = false;
	// continuous deviation from the reference
	#if defined (DOUBLE_FP)
	double max_delta;
	#else
	float max_delta;
	#endif
	// ndt parameters
	#if defined (DOUBLE_FP)
	double outlier_ratio_ = 0.55;
	float  resolution_    = 1.0;
	double trans_eps      = 0.01; //Transformation epsilon
	double step_size_     = 0.1;  // Step size
	#else
	float outlier_ratio_  = 0.55;
	float resolution_     = 1.0;
	float trans_eps       = 0.01; //Transformation epsilon
	float step_size_      = 0.1;  // Step size
	#endif

	int iter = 30;  // Maximum iterations
	Matrix4f final_transformation_, transformation_, previous_transformation_;
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

	#if defined (DOUBLE_FP)
	double gauss_d1_, gauss_d2_;
	double trans_probability_;
	double transformation_epsilon_ = 0.1;
	#else
	float gauss_d1_, gauss_d2_;
	float trans_probability_;
	float transformation_epsilon_ = 0.1;
	#endif
	int max_iterations_;

	// point clouds
	PointCloud* input_ = nullptr;
	PointCloud* target_ = nullptr;
	PointCloud* filtered_scan_ptr = nullptr;
	PointCloud* maps = nullptr;
	// voxel grid extends
	PointXYZI minVoxel, maxVoxel;
	int voxelDimension[3];
	// starting transformation matrix
	Matrix4f* init_guess = nullptr;
	// algorithm results
	CallbackResult* results = nullptr;
	// voxel grid spanning over the cloud
	ComputeEnv OCL_objs;
	cl::Buffer buff_target_cells;
	cl::Buffer buff_target;
	cl::Buffer buff_subvoxel;
	cl::Buffer buff_counter;

	cl::Kernel radiusSearchKernel;
	cl::Kernel findMinMaxKernel;
	cl::Kernel initTargetCellsKernel;
	cl::Kernel firstPassKernel;
	cl::Kernel secondPassKernel;

public:
	ndt_mapping();
public:
	virtual void init();
	virtual void quit();
	virtual void run(int p = 1);
	virtual bool check_output();
private:
	/**
	 * Reads the next reference matrix.
	 */
	void parseResult(std::ifstream& output_file, CallbackResult& result);
	/**
	 * Reads the next point cloud.
	 */
	void  parseFilteredScan(std::ifstream& input_file, PointCloud& pointcloud);
	/**
	 * Reads the next initilization matrix.
	 */
	void parseInitGuess(std::ifstream& input_file, Matrix4f& initGuess);
	/**
	 * Reads the number of testcases in the data file
	 */
	int read_number_testcases(std::ifstream& input_file);
	/**
	 * Reads the next testcases.
	 * count: number of datasets to read
	 * return: number of data sets actually read.
	 */
    int read_next_testcases(int count);
	/**
	 * Reads and compares algorithm results with the respective reference.
	 * count: number of testcase results to compare
	 */
	void check_next_outputs(int count);
	/**
	 * Reduces a multi dimensional voxel grid index to one dimension.
	 */
	inline int linearizeAddr(const int x, const int y, const int z);
	/**
	 * Reduces a coordinate to a voxel grid index.
	 */
	inline int linearizeCoord(const float x, const float y, const float z);

	#if defined (DOUBLE_FP)
	double updateDerivatives (Vec6 &score_gradient,
						Mat66 &hessian,
						Vec3 &x_trans, Mat33 &c_inv,
						bool compute_hessian = true);
	#else
	float updateDerivatives (Vec6 &score_gradient,
						Mat66 &hessian,
						Vec3 &x_trans, Mat33 &c_inv,
						bool compute_hessian = true);
	#endif

	void computePointDerivatives (Vec3 &x, bool compute_hessian = true);
	void computeHessian (Mat66 &hessian,
				PointCloudSource &trans_cloud, Vec6 &);
	void updateHessian (Mat66 &hessian, Vec3 &x_trans, Mat33 &c_inv);

	#if defined (DOUBLE_FP)
	double computeDerivatives (Vec6 &score_gradient,
						Mat66 &hessian,
						PointCloudSource &trans_cloud,
						Vec6 &p,
						bool compute_hessian = true );
	#else
	float computeDerivatives (Vec6 &score_gradient,
						Mat66 &hessian,
						PointCloudSource &trans_cloud,
						Vec6 &p,
						bool compute_hessian = true );
	#endif

	#if defined (DOUBLE_FP)
	bool updateIntervalMT (double &a_l, double &f_l, double &g_l,
					double &a_u, double &f_u, double &g_u,
					double a_t, double f_t, double g_t);
	double trialValueSelectionMT (double a_l, double f_l, double g_l,
							double a_u, double f_u, double g_u,
							double a_t, double f_t, double g_t);
	double computeStepLengthMT (const Vec6 &x, Vec6 &step_dir, double step_init, double step_max,
				double step_min, double &score, Vec6 &score_gradient, Mat66 &hessian,
				PointCloudSource &trans_cloud);
	#else
	bool updateIntervalMT (float &a_l, float &f_l, float &g_l,
					float &a_u, float &f_u, float &g_u,
					float a_t, float f_t, float g_t);
	float trialValueSelectionMT (float a_l, float f_l, float g_l,
							float a_u, float f_u, float g_u,
							float a_t, float f_t, float g_t);
	float computeStepLengthMT (const Vec6 &x, Vec6 &step_dir, float step_init, float step_max,
				float step_min, float &score, Vec6 &score_gradient, Mat66 &hessian,
				PointCloudSource &trans_cloud);
	#endif

	void computeTransformation(PointCloud &output, const Matrix4f &guess);
	void computeAngleDerivatives (Vec6 &p, bool compute_hessian = true);

	void ndt_align (const Matrix4f& guess);
	/**
	 * Performs point cloud specific voxel grid initialization.
	 */
	void initCompute();

	void buildTransformationMatrix(Matrix4f &matrix, Vec6 transform);

	#if defined (DOUBLE_FP)
	inline double ndt_getFitnessScore ();
	#else
	inline float ndt_getFitnessScore ();
	#endif
	/**
	 * Computes the eulerangles from an rotation matrix.
	 */
	void eulerAngles(Matrix4f transform, Vec3 &result);

	CallbackResult partial_points_callback(
		PointCloud &input_cloud,
		Matrix4f &init_guess,
		PointCloud& target_cloud
	);
};


#endif // EPHOS_NDT_MAPPING_H
