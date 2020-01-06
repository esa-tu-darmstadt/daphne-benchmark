.DEFAULT_GOAL := all
.PHONY: all clean ocl/device/ocl_kernel.h checkdata

-include Makefile.deps

CXXFLAGS=-O3
CXXFLAGS+= -std=c++11


# opencl library
OPENCL_INCLUDE_PATH=
OPENCL_LIBRARY_PATH=
ifdef OPENCL_INCLUDE_PATH
	CPPFLAGS+= -I$(OPENCL_INCLUDE_PATH)
endif
ifdef OPENCL_LIBRARY_PATH
	LDDFLAGS+= -L$(OPENCL_LIBRARY_PATH)
endif

# number of work items per work group
OPENCL_LOCAL_SIZE=512
CPPFLAGS+= -DNUMWORKITEMS_PER_WORKGROUP=$(OPENCL_LOCAL_SIZE)

# opencl platform
OPENCL_PLATFORM_ID=
OPENCL_DEVICE_ID=
OPENCL_DEVICE_TYPE=
ifdef OPENCL_PLATFORM_ID
	CPPFLAGS+= -DEPHOS_PLATFORM_HINT=$(OPENCL_PLATFORM_ID)
endif
ifdef OPENCL_DEVICE_ID
	CPPFLAGS+= -DEPHOS_DEVICE_HINT=$(OPENCL_DEVICE_ID)
endif
ifdef OPENCL_DEVICE_TYPE
	CPPFLAGS+= -DEPHOS_DEVICE_TYPE=$(OPENCL_DEVICE_TYPE)
endif

# opencl optimization
ifdef OPENCL_ZERO_COPY
	CPPFLAGS+= -DEPHOS_ZERO_COPY=1
endif
ifdef OPENCL_PINNED_MEMORY
	CPPFLAGS+= -DEPHOS_PINNED_MEMORY=1
endif
ifdef OPENCL_ATOMICS
	CPPFLAGS+= -DEPHOS_KERNEL_ATOMICS=1
endif

CPPFLAGS+= -I../include -I./
LDDFLAGS += -lOpenCL

all: kernel checkdata

kernel: ../common/main.o kernel.o ocl/host/ocl_tools.o framework.o
	$(CXX) $^ -o $@ $(LDDFLAGS)

%.o: %.cpp
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

ocl/device/ocl_kernel.h:
	./stringify_ocl_kernels.sh

checkdata:
ifeq ($(wildcard ../../../data/p2i_input.dat),)
	$(warning p2i_input.dat not found. Did you forget to extract the test data?)
endif
ifeq ($(wildcard ../../../data/p2i_output.dat),)
	$(warning p2i_output.dat not found. Did you forget to extract the test data?)
endif

clean:
	rm -f kernel kernel.o framework.o ../common/main.o Makefile.deps \
	ocl/host/ocl_tools.o ocl/device/ocl_kernel_tmp ocl/device/ocl_kernel.h

Makefile.deps:
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -MM -MG \
	../common/main.cpp kernel.cpp ocl/host/ocl_tools.cpp > Makefile.deps

