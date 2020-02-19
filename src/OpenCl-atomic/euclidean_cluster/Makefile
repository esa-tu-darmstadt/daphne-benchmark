.DEFAULT_GOAL := all
.PHONY: all clean checkdata kernel/kernel.h

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
OPENCL_WORK_GROUP_SIZE=512
CPPFLAGS+= -DEPHOS_KERNEL_WORK_GROUP_SIZE=$(OPENCL_WORK_GROUP_SIZE)
ifdef DISABLE_OPENCL_ATOMICS
$(info Disabled OpenCL atomic operations)
else
	CPPFLAGS+= -DEPHOS_KERNEL_ATOMICS
endif
ifdef DISABLE_OPENCL_LINE_PROCESSING
$(info Disabled line based processing in kernel)
else
	CPPFLAGS+= -DEPHOS_KERNEL_LINE_PROCESSING
endif
ifdef OPENCL_DISTANCE_PACKETS_PER_WORK_ITEM
$(info Processing $(OPENCL_DISTANCE_PACKETS_PER_WORK_ITEM) packets per work item)
	CPPFLAGS+= -DEPHOS_KERNEL_DISTANCE_PACKETS_PER_ITEM=$(OPENCL_DISTANCE_PACKETS_PER_WORK_ITEM)
endif
ifdef OPENCL_DISTANCES_PER_PACKET
$(info Working with packed distances of $(OPENCL_DISTANCES_PER_PACKET) bits)
	CPPFLAGS+= -DEPHOS_KERNEL_DISTANCES_PER_PACKET=$(OPENCL_DISTANCES_PER_PACKET)
endif

# opencl memory optimization
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
	CPPFLAGS+= -DTESTCASE_LIMIT=$(TESTCASE_LIMIT)
endif

all: euclidean_clustering checkdata

kernel/kernel.h:
	cd kernel && ./stringify_kernels.sh

euclidean_clustering: euclidean_clustering.o framework.o ../common/main.o ../common/compute_tools.o kernel/kernel.h
	$(CXX) $^ -o $@ $(LDDFLAGS)

%.o: %.cpp
	$(CXX) -c $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -o $@ $<

checkdata:
ifeq ($(wildcard ../../../data/ec_input.dat),)
	$(warning ec_input.dat not found. Did you forget to extract the test data?)
endif
ifeq ($(wildcard ../../../data/ec_output.dat),)
	$(warning ec_output.dat not found. Did you forget to extract the test data?)
endif

clean:
	rm -f Makefile.deps euclidean_clustering euclidean_clustering.o framework.o \
	../common/main.o ../common/compute_tools.o kernel/kernel.h

Makefile.deps:
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -I../include -MM -MG \
	../common/main.cpp ../common/compute_tools.cpp euclidean_clustering.cpp framework.cpp > Makefile.deps


