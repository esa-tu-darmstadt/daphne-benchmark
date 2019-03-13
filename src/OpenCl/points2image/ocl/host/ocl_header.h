#ifndef ocl_ephos_h
#define ocl_ephos_h

// OpenCL C++ headers

// Added to avoid "warning: <CL-API> is deprecated" compiler messages
// clinfo tells that device support v1.2, 
// although platform supports v2.1

#define CL_TARGET_OPENCL_VERSION 120
//#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#define __CL_ENABLE_EXCEPTIONS
//#include <CL/cl.hpp>
#include <CL/cl.h>

// Compute c = a + b.
static const char source[] =
    "#if defined(cl_khr_fp64)\n"
    "#  pragma OPENCL EXTENSION cl_khr_fp64: enable\n"
    "#elif defined(cl_amd_fp64)\n"
    "#  pragma OPENCL EXTENSION cl_amd_fp64: enable\n"
    "#else\n"
    "#  error double precision is not supported\n"
    "#endif\n"
    "kernel void add(\n"
    "       ulong n,\n"
    "       global const double *a,\n"
    "       global const double *b,\n"
    "       global double *c\n"
    "       )\n"
    "{\n"
    "    size_t i = get_global_id(0);\n"
    "    if (i < n) {\n"
    "       c[i] = a[i] + b[i];\n"
    "    }\n"
    "}\n";

// Struct for passing OpenCL objects
struct OCL_Struct {
#if 0
	  cl::Platform     platform;
	  cl::Device       device;
	  cl::Context      context;
	  cl::CommandQueue cmdqueue;
#endif
	cl_platform_id   *rcar_platform;
	cl_device_id     cvengine_device;
	cl_context       rcar_context;
	cl_command_queue cvengine_command_queue;
};

// SP: single precision flavors of structs
typedef struct SP_Mat44 {
  float data[4][4];
} SP_Mat44;

typedef struct SP_Mat33 {
  float data[3][3];
} SP_Mat33;

typedef struct SP_Mat13 {
  float data[3];
} SP_Mat13;

typedef struct SP_Vec5 {
  float data[5];
} SP_Vec5;

typedef struct SP_Point2d {
  float x;
  float y;
} SP_Point2d;




#endif

