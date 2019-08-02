/**
 * Author:  Leonardo Solis, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2018
 * Author:  Thilo Gabel, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2019
 * License: Apache 2.0 (see attachached File)
 */
#ifndef OCL_EPHOS_H
#define OCL_EPHOS_H

#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.h>

// Struct for passing OpenCL objects
struct OCL_Struct {
	cl_device_id     device;
	cl_context       context;
	cl_command_queue cmdqueue;
};

class OCL_Tools {
public:
	/**
	 * Searches through the available OpenCL platforms to find one that suits the given arguments.
	 * platformHint: platform name or index, empty for no restriction
	 * deviceHint: device name or index, empty for no restriction
	 * deviceType: can be one of ALL, CPU, GPU, ACC, DEF to only allow certaind devices
	 * extensions: a chosen device must support at least one extension from each given extension set
	 * return: platform objects
	 */
	static OCL_Struct find_compute_platform(
		std::string platformHint, std::string deviceHint, std::string deviceType,
		std::vector<std::vector<std::string>> extensions);
	/**
	 * Builds an OpenCL program and extracts the requested kernels.
	 * ocl_objs: platform oobjects
	 * sources: program source code
	 * options: program build arguments
	 * kernelNames: kernels to extract
	 * kernels: empty list of extracted kernels
	 * return: the program build from source
	 */
	static cl_program build_program(OCL_Struct& ocl_objs, std::string& sources,
		std::string options, std::vector<std::string>& kernelNames, std::vector<cl_kernel>& kernels);
};
#endif

