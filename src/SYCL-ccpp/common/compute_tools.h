/**
 * Author:  Thilo Gabel, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2019
 * License: Apache 2.0 (see attachached File)
 */
#ifndef EPHOS_COMPUTE_TOOLS_H
#define EPHOS_COMPUTE_TOOLS_H

#include <iostream>
#include <SYCL/sycl.hpp>


typedef struct ComputeEnv {
	cl::sycl::device device;
	cl::sycl::queue cmdqueue;
} ComputeEnv;

class ComputeTools {
public:
	static ComputeEnv find_compute_platform(const std::string& platformHint, const std::string& deviceHint,
		const std::string& deviceType);
	//static cl::sycl::device findComputeDevice(const std::string& deviceType);
	//static cl::sycl::device selectComputeDevice(
	//	const std::string& platformHint, const std::string& deviceHint, const std::string& deviceTypeHint);
};

#endif
