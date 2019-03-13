#ifndef ocl_ephos_h
#define ocl_ephos_h

// OpenCL C++ headers

// Added to avoid "warning: <CL-API> is deprecated" compiler messages
// clinfo tells that device support v1.2,
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#if defined (OPENCL_CPP_WRAPPER)
	#define __CL_ENABLE_EXCEPTIONS
	#include <CL/cl.hpp>
#else
	#include <CL/cl.h>
#endif

// Struct for passing OpenCL objects
struct OCL_Struct {

	#if defined (OPENCL_CPP_WRAPPER)
	//cl::Platform     platform;
	cl::Device       device;
	cl::Context      context;
	cl::CommandQueue cmdqueue;
	cl::Kernel       kernel_initRS;
	cl::Kernel       kernel_parallelRS;
	#else
	cl_platform_id   *rcar_platform;
	cl_device_id     cvengine_device;
	cl_context       rcar_context;
	cl_command_queue cvengine_command_queue;
        cl_kernel        kernel_initRS;
        cl_kernel        kernel_parallelRS;  
	#endif
};

#endif


