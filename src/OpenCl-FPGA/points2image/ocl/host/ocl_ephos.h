/**
 * Author:  Leonardo Solis, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2018
 * License: Apache 2.0 (see attachached File)
 */
#ifndef ocl_ephos_h
#define ocl_ephos_h

//OpenCL utility layer include
#include "xcl2.hpp"

// Struct for passing OpenCL objects
struct Struct_OCLEphos {
    cl::Context      context;
    cl::CommandQueue q;
    cl::Kernel       kernel;
};

#endif
