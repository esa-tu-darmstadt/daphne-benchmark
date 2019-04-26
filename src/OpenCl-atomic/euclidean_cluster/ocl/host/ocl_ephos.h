#ifndef OCL_EPHOS_H
#define OCL_EPHOS_H

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_EXCEPTIONS
// backwards compatibility
#define __CL_ENABLE_EXCEPTIONS
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY

#include <CL/cl.hpp>

struct OCL_Struct {
	cl::Device       device;
	cl::Context      context;
	cl::CommandQueue cmdqueue;
	cl::Kernel       kernel_initRS;
	cl::Kernel       kernel_parallelRS;
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
	static cl::Program build_program(OCL_Struct& ocl_objs, cl::Program::Sources& sources,
		std::string options, std::vector<std::string>& kernelNames, std::vector<cl::Kernel>& kernels);
};
#endif


