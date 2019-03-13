.DEFAULT_GOAL := kernel

-include Makefile.deps

CPPFLAGS = -std=c++11

DEBUG=0

ifeq ($(DEBUG),1)
	CPPFLAGS += -g -Wall
else
	CPPFLAGS +=  -O3
endif

# Raven Ridge
#OPENCL_INCLUDE_PATH=/opt/rocm/opencl/include
#OPENCL_LIBRARY_PATH=/opt/rocm/opencl/lib/x86_64

# Using C++ wrapper
#OPENCL_INCLUDE_PATH=/home/root/CLHPP-headers/OpenCL-CLHPP/build/include
#OPENCL_LIBRARY_PATH=/home/root/ComputeSuite-CE-0.1.0-Yocto-2.23.1-RCAR/V3M2/ComputeAorta/lib/

# Renesas (after update)
OPENCL_INCLUDE_PATH=/home/root/CL-headers
#OPENCL_LIBRARY_PATH=../../../../RCar-CE-0.3.0-Yocto-2.23.1-Release/ComputeAorta/V3M2/lib/

# Renesas (further updates reuse same CL-headers)
# IMPORTANT: verify that LD_LIBRARY_PATH was set with the correct lib version!
#            check /home/root/EPHoS/sourcemeopencl<X>.sh

#OPENCL_LIBRARY_PATH=../../../../RCar-CE-0.3.1-Yocto-2.23.1-Release/ComputeAorta/V3M2/lib/

#OPENCL_LIBRARY_PATH=../../../../RCar-CE-0.3.2-Yocto-2.23.1-Release/ComputeAorta/V3M2/lib/
OPENCL_LIBRARY_PATH==/home/root/CE-v3m2/lib/


PRINTINFO=0

ifeq ($(PRINTINFO),1)
        CPPFLAGS += -DPRINTINFO
else
        CPPFLAGS +=
endif

OPENCL=0

ifeq ($(OPENCL),1)
	CPPFLAGS += -I${OPENCL_INCLUDE_PATH} -I./ocl/host -I./  -DOPENCL
	LDDFLAGS += -L$(OPENCL_LIBRARY_PATH) -lOpenCL 
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

