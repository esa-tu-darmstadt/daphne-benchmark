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
#include <vector>

#include "datatypes.h"
#include "common/ndt_mapping_base.h"
#include "common/compute_tools.h"


#define STRINGIFY2(s) #s
#define STRINGIFY(s) STRINGIFY2(s)

#define EPHOS_KERNEL_WORK_GROUP_SIZE_S STRINGIFY(EPHOS_KERNEL_WORK_GROUP_SIZE)

// maximum allowed deviation from reference
#define MAX_TRANSLATION_EPS 0.001
#define MAX_ROTATION_EPS 1.8
#define MAX_EPS 2

// opencl platform hints
#if defined(EPHOS_PLATFORM_HINT)
#define EPHOS_PLATFORM_HINT_S STRINGIFY(EPHOS_PLATFORM_HINT)
#else
#define EPHOS_PLATFORM_HINT_S ""
#endif

#if defined(EPHOS_DEVICE_HINT)
#define EPHOS_DEVICE_HINT_S STRINGIFY(EPHOS_DEVICE_HINT)
#else
#define EPHOS_DEVICE_HINT_S ""
#endif

#if defined(EPHOS_DEVICE_TYPE)
#define EPHOS_DEVICE_TYPE_S STRINGIFY(EPHOS_DEVICE_TYPE)
#else
#define EPHOS_DEVICE_TYPE_S ""
#endif



class ndt_mapping : public ndt_mapping_base {
private:
	// compute members
	ComputeEnv computeEnv;
	cl::Buffer voxelGridBuffer;
	cl::Buffer pointCloudBuffer;
	cl::Buffer subvoxelBuffer;
	cl::Buffer counterBuffer;
	cl::Buffer gridInfoBuffer;
#ifdef EPHOS_PINNED_MEMORY
	cl::Buffer subvoxelHostBuffer;
	PointVoxel* subvoxelStorage;
#elif defined(EPHOS_ZERO_COPY)
#else
	PointVoxel* subvoxelStorage;
#endif

	cl::Kernel measureCloudKernel;
	cl::Kernel initTargetCellsKernel;
	cl::Kernel firstPassKernel;
	cl::Kernel secondPassKernel;
	cl::Kernel radiusSearchKernel;

	size_t maxComputeGridSize;
	size_t maxComputeCloudSize;

public:
	ndt_mapping();
	virtual ~ndt_mapping();
public:
	virtual void init();
	virtual void quit();
private:
	double updateDerivatives (Vec6 &score_gradient,
		Mat66 &hessian,
		Vec3 &x_trans, Mat33 &c_inv,
		bool compute_hessian = true);

	void computePointDerivatives (Vec3 &x, bool compute_hessian = true);

	void computeHessian (Mat66 &hessian, PointCloudSource &trans_cloud, Vec6 &);

	void updateHessian (Mat66 &hessian, Vec3 &x_trans, Mat33 &c_inv);

	double computeDerivatives (Vec6 &score_gradient,
		Mat66 &hessian,
		PointCloudSource &trans_cloud,
		Vec6 &p,
		bool compute_hessian = true );


	void computeAngleDerivatives (Vec6 &p, bool compute_hessian = true);
	/**
	 * Performs point cloud specific voxel grid initialization.
	 */
	void initCompute();

	/**
	 * Initializes compute buffers for the next iteration.
	 */
	void prepare_compute_buffers(int cloudSize, int* gridSize);

	int pack_minmaxf(float val);

	float unpack_minmaxf(int val);
};


#endif // EPHOS_NDT_MAPPING_H
