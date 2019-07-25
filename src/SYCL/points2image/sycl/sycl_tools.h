#ifndef EPHOS_SYCL_TOOLS_H
#define EPHOS_SYCL_TOOLS_H

#include <iostream>
#include <SYCL/sycl.hpp>

class SyclTools {
public:
	static cl::sycl::device findComputeDevice(const std::string& deviceType); 
};

#endif
