#ifndef OCL_EPHOS_H
#define OCL_EPHOS_H

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

struct OCL_Struct {
	cl::Device       device;
	cl::Context      context;
	cl::CommandQueue cmdqueue;
	cl::Kernel       kernel_initRS;
	cl::Kernel       kernel_parallelRS;
};

#endif


