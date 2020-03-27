.DEFAULT_GOAL := all
.PHONY: all clean kernel/kernel.h checkdata

-include Makefile.deps

ifdef DEBUG
	CXXFLAGS+= -g
else
	CXXFLAGS=-O3
endif
CXXFLAGS+= -std=c++11
CPPFLAGS+= -I./ -I../

# opencl headers and library
ifdef OPENCL_INCLUDE_PATH
	CPPFLAGS+= -I${OPENCL_INCLUDE_PATH}
endif
ifdef OPENCL_LIBRARY_PATH
	LDDFLAGS+= -L$(OPENCL_LIBRARY_PATH)
endif
LDDFLAGS+= -lOpenCL

# opencl platform
ifdef OPENCL_PLATFORM_ID
$(info Searching for OpenCL platform $(OPENCL_PLATFORM_ID))
	CPPFLAGS+= -DEPHOS_PLATFORM_HINT=$(OPENCL_PLATFORM_ID)
endif
ifdef OPENCL_DEVICE_ID
$(info Searching for OpenCL device $(OPENCL_DEVICE_ID))
	CPPFLAGS+= -DEPHOS_DEVICE_HINT=$(OPENCL_DEVICE_ID)
endif
ifdef OPENCL_DEVICE_TYPE
$(info Searching for OpenCL device type $(OPENCL_DEVICE_TYPE))
	CPPFLAGS+= -DEPHOS_DEVICE_TYPE=$(OPENCL_DEVICE_TYPE)
endif

# opencl kernel optimization
OPENCL_WORK_GROUP_SIZE=256
CPPFLAGS+= -DEPHOS_KERNEL_WORK_GROUP_SIZE=$(OPENCL_WORK_GROUP_SIZE)
ifdef ENABLE_OPENCL_CLOUD_MEASURE
$(info Enabled OpenCL cloud measurements)
	CPPFLAGS+= -DEPHOS_KERNEL_CLOUD_MEASURE
endif
ifdef OPENCL_VOXEL_INTERNAL_POINT_BUFFER
$(info Enabled voxel internal point buffers of size $(OPENCL_VOXEL_INTERNAL_POINT_BUFFER))
	CPPFLAGS+= -DEPHOS_KERNEL_VOXEL_INTERNAL_POINTS=$(OPENCL_VOXEL_INTERNAL_POINT_BUFFER)
endif

# opencl memory optimization
ifdef ENABLE_OPENCL_ZERO_COPY
$(info Enabled OpenCL zero copy)
	CPPFLAGS+= -DEPHOS_ZERO_COPY
endif
ifdef ENABLE_OPENCL_PINNED_MEMORY
$(info Enabled OpenCL page locked memory)
	CPPFLAGS+= -DEPHOS_PINNED_MEMORY
endif



# other options
ifdef TESTCASE_LIMIT
	CPPFLAGS+= -DEPHOS_TESTCASE_LIMIT=$(TESTCASE_LIMIT)
endif
ifdef ENABLE_DATAGEN
	CPPFLAGS+= -DEPHOS_DATAGEN
endif

all: benchmark checkdata

kernel/kernel.h:
	cd kernel && ./stringify_kernels.sh

benchmark: ndt_mapping.o framework.o ../common/main.o ../common/compute_tools.o
	$(CXX) $^ -o $@ $(LDDFLAGS)

%.o: %.cpp
	$(CXX) -c $(CLFAGS) $(CPPFLAGS) $(CXXFLAGS) -o $@ $<

checkdata:
ifeq ($(wildcard ../../../data/ndt_input.dat),)
	$(warning ndt_input.dat not found. Did you forget to extract the test data?)
endif
ifeq ($(wildcard ../../../data/ndt_output.dat),)
	$(warning ndt_output.dat not found. Did you forget to extract the test data?)
endif

clean:
	rm -f Makefile.deps benchmark ndt_mapping.o ../common/main.o framework.o \
	../common/main.o ../common/compute_tools.o kernel/kernel.h

Makefile.deps:
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -MM -MG \
	../common/main.cpp ../common/compute_tools.cpp ndt_mapping.cpp framework.cpp > Makefile.deps
