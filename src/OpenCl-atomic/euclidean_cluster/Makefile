.DEFAULT_GOAL := all
.PHONY: all clean checkdata kernel/kernel.h

-include Makefile.deps

ifdef DEBUG
	CXXFLAGS+= -g
else
	CXXFLAGS+= -O3
endif
CXXFLAGS+= -std=c++11
CPPFLAGS+= -I../ -I../../base/benchmark -I../../base/euclidean_cluster

# opencl headers and library
ifdef OPENCL_INCLUDE_PATH
	CPPFLAGS+= -I$(OPENCL_INCLUDE_PATH)
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

ifdef DISABLE_OPENCL_LINE_PROCESSING
$(info Disabled line based processing in kernel)
else
	CPPFLAGS+= -DEPHOS_KERNEL_LINE_PROCESSING
endif
OPENCL_DISTANCE_PACKETS_PER_WORK_ITEM=1
OPENCL_DISTANCES_PER_PACKET=8
ifneq ($(OPENCL_DISTANCES_PER_PACKET),8)
$(info Working with distance packets with $(OPENCL_DISTANCES_PER_PACKET) elements)
endif
ifneq ($(OPENCL_DISTANCE_PACKETS_PER_WORK_ITEM),1)
$(info Processing $(OPENCL_DISTANCE_PACKETS_PER_WORK_ITEM) packets with each work item)
endif
CPPFLAGS+= -DEPHOS_KERNEL_DISTANCE_PACKETS_PER_ITEM=$(OPENCL_DISTANCE_PACKETS_PER_WORK_ITEM)
CPPFLAGS+= -DEPHOS_KERNEL_DISTANCES_PER_PACKET=$(OPENCL_DISTANCES_PER_PACKET)

# opencl memory optimization
ifdef DISABLE_OPENCL_ATOMICS
$(info Disabled OpenCL atomic operations)
else
	CPPFLAGS+= -DEPHOS_KERNEL_ATOMICS
endif
ifdef ENABLE_OPENCL_ZERO_COPY
$(info Enabled OpenCL zero copy memory mode)
	CPPFLAGS+= -DEPHOS_ZERO_COPY
endif
ifdef ENABLE_OPENCL_PINNED_MEMORY
$(info Enabled OpenCL page locked memory)
	CPPFLAGS+= -DEPHOS_PINNED_MEMORY
endif

# other options
ifdef TESTCASE_LIMIT
$(info Limiting testcases to $(TESTCASE_LIMIT))
	CPPFLAGS+= -DEPHOS_TESTCASE_LIMIT=$(TESTCASE_LIMIT)
endif
ifdef ENABLE_TESTDATA_GEN
$(info Enabled result data generation)
	CPPFLAGS+= -DEPHOS_TESTDATA_GEN
endif

all: benchmark checkdata

benchmark: euclidean_clustering.o euclidean_clustering_base.o main.o compute_tools.o
	$(CXX) $^ -o $@ $(LDDFLAGS)

kernel/kernel.h:
	cd kernel && ./stringify_kernels.sh

%.o: %.cpp
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

%.o: ../../base/benchmark/common/%.cpp
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

%.o: ../../base/euclidean_cluster/common/%.cpp
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

%.o: ../common/%.cpp
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

checkdata:
ifeq ($(wildcard ../../../data/ec_input.dat),)
	$(warning ec_input.dat not found. Did you forget to extract the test data?)
endif
ifeq ($(wildcard ../../../data/ec_output.dat),)
	$(warning ec_output.dat not found. Did you forget to extract the test data?)
endif

clean:
	rm -f Makefile.deps benchmark euclidean_clustering.o euclidean_clustering_base.o \
	main.o compute_tools.o kernel/kernel.h

Makefile.deps:
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS)  -MM -MG \
	../../base/benchmark/common/main.cpp ../common/compute_tools.cpp euclidean_clustering.cpp \
	../../base/euclidean_cluster/common/euclidean_clustering_base.cpp > Makefile.deps


