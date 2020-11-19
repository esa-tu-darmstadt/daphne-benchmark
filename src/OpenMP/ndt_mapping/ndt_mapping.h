/**
 * Author:  Florian Stock, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2018
 * Author:  Thilo Gabel, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2019 - 2020
 * License: Apache 2.0 (see attached files)
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

#include "common/ndt_mapping_base.h"

// maximum allowed deviation from reference
#define MAX_TRANSLATION_EPS 0.001
#define MAX_ROTATION_EPS 1.8
#define MAX_EPS 2

class ndt_mapping : public ndt_mapping_base {
public:
	ndt_mapping();
	virtual ~ndt_mapping();
public:
	virtual void init();
	virtual void quit();
private:
	int voxelRadiusSearch(VoxelGrid& grid, const PointXYZI& point,
		double radius, std::vector<Voxel*>& indices);

	double updateDerivatives(Vec6& score_gradient,
		Mat66& hessian,
		Vec3& x_trans, Mat33& c_inv,
		bool compute_hessian=true);

	void computePointDerivatives(Vec3 &x, bool compute_hessian = true);
	void computeHessian(Mat66& hessian, PointCloud& trans_cloud, Vec6&);
	void updateHessian(Mat66& hessian, Vec3& x_trans, Mat33& c_inv);

	double computeDerivatives(Vec6& score_gradient,
		Mat66& hessian,
		PointCloud& trans_cloud,
		Vec6& p,
		bool compute_hessian=true);


	void computeAngleDerivatives(Vec6& p, bool compute_hessian=true);

	virtual void initCompute();
	virtual void cleanupCompute();

};


#endif // EPHOS_NDT_MAPPING_H
