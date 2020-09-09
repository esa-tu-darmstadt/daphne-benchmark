.DEFAULT_GOAL := all
.PHONY: all clean kernel/kernel.h checkdata

-include Makefile.deps

ifdef DEBUG
	CXXFLAGS+= -g
else
	CXXFLAGS=-O3
endif
CXXFLAGS+= -std=c++11
CPPFLAGS+= -I.. -I../../base/benchmark -I../../base/ndt_mapping

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
ifdef OPENCL_VOXEL_POINT_STORAGE
$(info Enabled voxel internal point storage of size $(OPENCL_VOXEL_POINT_STORAGE))
	CPPFLAGS+= -DEPHOS_KERNEL_VOXEL_POINT_STORAGE=$(OPENCL_VOXEL_POINT_STORAGE)
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

benchmark: ndt_mapping.o ndt_mapping_base.o main.o compute_tools.o
	$(CXX) $^ -o $@ $(LDDFLAGS)

%.o: %.cpp
	$(CXX) $(CLFAGS) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

%.o: ../common/%.cpp
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

%.o: ../../base/benchmark/common/%.cpp
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

%.o: ../../base/ndt_mapping/common/%.cpp
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

checkdata:
ifeq ($(wildcard ../../../data/ndt_input.dat),)
	$(warning ndt_input.dat not found. Did you forget to extract the test data?)
endif
ifeq ($(wildcard ../../../data/ndt_output.dat),)
	$(warning ndt_output.dat not found. Did you forget to extract the test data?)
endif

clean:
	rm -f Makefile.deps benchmark ndt_mapping.o ndt_mapping_base.o main.o \
	compute_tools.o kernel/kernel.h

Makefile.deps:
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -MM -MG \
	../../base/benchmark/common/main.cpp ../common/compute_tools.cpp \
	../../base/ndt_mapping/common/ndt_mapping_base.cpp ndt_mapping.cpp > Makefile.deps
