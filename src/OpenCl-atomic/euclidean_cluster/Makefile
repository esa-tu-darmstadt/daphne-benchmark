.DEFAULT_GOAL := all
.PHONY: all clean ocl/device/ocl_kernel.h checkdata

-include Makefile.deps

CXXFLAGS=-O3
CXXFLAGS+= -std=c++11

# opencl headers and library
OPENCL_INCLUDE_PATH=
OPENCL_LIBRARY_PATH=
ifneq ($(OPENCL_INCLUDE_PATH),)
	CPPFLAGS+= -I$(OPENCL_INCLUDE_PATH)
endif
CPPFLAGS+= -I./ -I../
ifneq ($(OPENCL_LIBRARY_PATH),)
	LDDFLAGS+= -L$(OPENCL_LIBRARY_PATH)
endif
LDDFLAGS+= -lOpenCL

# number of work items per work group
OPENCL_LOCAL_SIZE=256
CPPFLAGS+= -DNUMWORKITEMS_PER_WORKGROUP=$(OPENCL_LOCAL_SIZE)

# double precision floating point
CPPFLAGS+= -DDOUBLE_FP

# opencl platform
OPENCL_PLATFORM_ID=
OPENCL_DEVICE_ID=
OPENCL_DEVICE_TYPE=
ifneq ($(OPENCL_PLATFORM_ID),)
	CPPFLAGS+= -DEPHOS_PLATFORM_HINT=$(OPENCL_PLATFORM_ID)
endif
ifneq ($(OPENCL_DEVICE_ID),)
	CPPFLAGS+= -DEPHOS_DEVICE_HINT=$(OPENCL_DEVICE_ID)
endif
ifneq ($(OPENCL_DEVICE_TYPE),)
	CPPFLAGS+= -DEPHOS_DEVICE_TYPE=$(OPENCL_DEVICE_TYPE)
endif

all: euclidean_clustering checkdata

ocl/device/ocl_kernel.h:
	./stringify_ocl_kernels.sh

euclidean_clustering: euclidean_clustering.o framework.o ../common/main.o ../common/compute_tools.o
	$(CXX) $^ -o $@ $(LDDFLAGS)

%.o: %.cpp
	$(CXX) -c $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -o $@ $<

#../common/main.o: ../common/main.cpp
#	$(CXX) -c $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -I../include $< -o $@

#kernel.o: kernel.cpp
#	$(CXX) -c $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -I../include $< -o $@

#ocl/host/ocl_tools.o: ocl/host/ocl_tools.cpp
#	$(CXX) -c $(CLFAGS) $(CPPFLAGS) $(CXXFLAGS) -I../include $< -o $@

checkdata:
ifeq ($(wildcard ../../../data/ec_input.dat),)
	$(warning ec_input.dat not found. Did you forget to extract the test data?)
endif
ifeq ($(wildcard ../../../data/ec_output.dat),)
	$(warning ec_output.dat not found. Did you forget to extract the test data?)
endif

clean:
	rm -f Makefile.deps euclidean_clustering euclidean_clustering.o framework.o \
	../common/main.o ../common/compute_tools.o ocl/host/ocl_tools.o ocl/device/ocl_kernel_tmp ocl/device/ocl_kernel.h

Makefile.deps:
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -I../include -MM -MG \
	../common/main.cpp euclidean_clustering.cpp framework.cpp ocl/host/ocl_tools.cpp > Makefile.deps


