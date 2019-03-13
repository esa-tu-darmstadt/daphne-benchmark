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
