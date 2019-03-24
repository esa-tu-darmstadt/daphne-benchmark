#ifndef OCL_EPHOS_H
#define OCL_EPHOS_H

// OpenCL C++ headers
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>


// Struct for passing OpenCL objects
struct OCL_Struct {
	cl::Device       device;
	cl::Context      context;
	cl::CommandQueue cmdqueue;
	cl::Kernel       kernel_findMinMax;
        cl::Kernel       kernel_initTargetCells;
	cl::Kernel       kernel_firstPass;
	cl::Kernel       kernel_secondPass;
};

#endif



