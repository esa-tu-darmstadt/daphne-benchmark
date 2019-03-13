.DEFAULT_GOAL := kernel

-include Makefile.deps

CPPFLAGS = -std=c++11

# DEBUG capabilities in host
DEBUG=0

ifeq ($(DEBUG),1)
	CPPFLAGS += -g -Wall -O0
else
	CPPFLAGS += -O3
endif

# Raven Ridge
#OPENCL_INCLUDE_PATH=/opt/rocm/opencl/include
#OPENCL_LIBRARY_PATH=/opt/rocm/opencl/lib/x86_64

# Using C++ wrapper
OPENCL_INCLUDE_PATH=/home/root/CLHPP-headers/OpenCL-CLHPP/build/include
#OPENCL_LIBRARY_PATH=/home/root/ComputeSuite-CE-0.1.0-Yocto-2.23.1-RCAR/V3M2/ComputeAorta/lib/

# Renesas (after update)
#OPENCL_INCLUDE_PATH=/home/root/CL-headers
#OPENCL_LIBRARY_PATH=../../../../RCar-CE-0.3.0-Yocto-2.23.1-Release/ComputeAorta/V3M2/lib/

# Renesas (further updates reuse same CL-headers)
# IMPORTANT: verify that LD_LIBRARY_PATH was set with the correct lib version!
#            check /home/root/EPHoS/sourcemeopencl<X>.sh

#OPENCL_LIBRARY_PATH=../../../../RCar-CE-0.3.1-Yocto-2.23.1-Release/ComputeAorta/V3M2/lib/
#OPENCL_LIBRARY_PATH=../../../../RCar-CE-0.3.2-Yocto-2.23.1-Release/ComputeAorta/V3M2/lib/
OPENCL_LIBRARY_PATH=/home/root/CE-v3m2/lib/

# Vega56 (in eredmithrim)
#OPENCL_INCLUDE_PATH=/home/lvs/ESA/AMDAPPSDK-3.0/include/
#OPENCL_LIBRARY_PATH=/opt/amdgpu-pro/lib/x86_64-linux-gnu

# Nvidia Jetson (POCL)
# Don't forget: export LD_LIBRARY_PATH=/scratch/user/local/lib
#OPENCL_INCLUDE_PATH=/scratch/user/local/include
#OPENCL_LIBRARY_PATH=/scratch/user/local/lib

# Printing information
PRINTINFO=0

ifeq ($(PRINTINFO),1)
        CPPFLAGS += -DPRINTINFO
else
        CPPFLAGS +=
endif

# OpenCL support
# Enabled by default because disabling it is NOT equivalent to serial version
OPENCL=1

# OpenCL C++ wrapper
# Enabled by default, otherwise plain OpenCL C APIs are used
OPENCL_CPP_WRAPPER = 1

# Local size: # work-items per work group
LOCAL_SIZE=256

ifeq ($(OPENCL),1)
	CPPFLAGS += -I${OPENCL_INCLUDE_PATH} -I./ocl/host -I./ -DOPENCL_EPHOS -DNUMWORKITEMS_PER_WORKGROUP=${LOCAL_SIZE}
	LDDFLAGS += -L$(OPENCL_LIBRARY_PATH) -lOpenCL

	ifeq ($(OPENCL_CPP_WRAPPER),1)
	CPPFLAGS += -DOPENCL_CPP_WRAPPER 
	endif
else
	CPPFLAGS +=
	LDDFLAGS +=
endif

# Double precision floating point
# Utilized in host and device
DOUBLE_FP=1

ifeq ($(DOUBLE_FP),1)
        CPPFLAGS += -DDOUBLE_FP
else
        CPPFLAGS +=
endif

stringify:
	./stringify_ocl_krnls.sh

kernel: ../common/main.o kernel.o 
	$(CXX) $^ -o $@ $(LDDFLAGS)

../common/main.o: ../common/main.cpp
	$(CXX) -c $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -I../include $< -o $@

kernel.o: kernel.cpp stringify
	$(CXX) -c $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -I../include $< -o $@

.PHONY: clean
clean:
	rm -f kernel kernel.o ../common/main.o Makefile.deps

Makefile.deps:
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -I../include -MM ../common/main.cpp *.cpp > Makefile.deps

