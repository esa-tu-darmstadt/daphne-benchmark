/**
 * Author:  Thilo Gabel, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2019
 * License: Apache 2.0 (see attachached File)
 */
#include <sycl/sycl_tools.h>

#include <CL/sycl.hpp>

cl::sycl::device SyclTools::findComputeDevice(const std::string& deviceType) {
	cl::sycl::device result;
	if (deviceType.compare("CPU") == 0) {
		result = cl::sycl::cpu_selector().select_device();
	} else if (deviceType.compare("GPU") == 0) {
		result = cl::sycl::gpu_selector().select_device();
	} else if (deviceType.compare("HOST") == 0) {
		result = cl::sycl::host_selector().select_device();
	} else {
		result = cl::sycl::host_selector().select_device();
	}
	return result;
}
