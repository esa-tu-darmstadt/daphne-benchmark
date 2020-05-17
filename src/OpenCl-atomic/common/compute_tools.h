/**
 * Author:  Leonardo Solis, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2018
 * Author:  Thilo Gabel, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2019
 * License: Apache 2.0 (see attachached File)
 */
#ifndef EPHOS_COMPUTE_TOOLS_H
#define EPHOS_COMPUTE_TOOLS_H

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#define CL_HPP_ENABLE_EXCEPTIONS

#define __CL_ENABLE_EXCEPTIONS
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY

#include <CL/cl.hpp>
#include <string>
#include <vector>

// Struct for passing OpenCL objects
typedef struct ComputeEnv {
	cl::Platform platform;
	cl::Device device;
	cl::Context context;
	cl::CommandQueue cmdqueue;
} ComputeEnv;

class ComputeTools {
public:
	/**
	 * Searches through the available OpenCL platforms to find one that suits the given arguments.
	 * platformHint: platform name or index, empty for no restriction
	 * deviceHint: device name or index, empty for no restriction
	 * deviceType: can be one of ALL, CPU, GPU, ACC, DEF to only allow certaind devices
	 * extensions: a chosen device must support at least one extension from each given extension set
	 * return: platform objects
	 */
	static ComputeEnv find_compute_platform(
		std::string platformHint, std::string deviceHint, std::string deviceType,
		std::vector<std::vector<std::string>> extensions);
	/**
	 * Builds an OpenCL program and extracts the requested kernels.
	 * computeEnv: platform oobjects
	 * sources: program source code
	 * options: program build arguments
	 * kernelNames: kernels to extract
	 * kernels: output kernel list
	 * return: the program build from source
	 */
	static cl::Program build_program(ComputeEnv& computeEnv, std::string& sources,
		std::string options, std::vector<std::string>& kernelNames, std::vector<cl::Kernel>& kernels);
};
#endif // EPHOS_COMPUTE_TOOLS_H

