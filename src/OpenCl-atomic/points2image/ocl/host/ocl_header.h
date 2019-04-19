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
#endif

