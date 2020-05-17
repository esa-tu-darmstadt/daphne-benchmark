.DEFAULT_GOAL := all
.PHONY: all clean checkdata kernel/kernel.h

-include Makefile.deps

ifdef DEBUG
	CXXFLAGS+= -g
else
	CXXFLAGS+= -O3
endif
CXXFLAGS+= -std=c++11
CPPFLAGS+= -I./ -I../

# standard optimization
CLOUD_DISTANCES_PER_PACKET=8
ifneq ($(CLOUD_DISTANCES_PER_PACKET),8)
$(info Working with distance packets with $(CLOUD_DISTANCES_PER_PACKET) elements)
endif
CPPFLAGS+= -DEPHOS_CLOUD_DISTANCES_PER_PACKET=$(CLOUD_DISTANCES_PER_PACKET)

# sycl platform flags
ifdef SYCL_DEVICE_TYPE
	CPPFLAGS+= -DEPHOS_DEVICE_TYPE=$(SYCL_DEVICE_TYPE)
endif
ifdef SYCL_DEVICE_NAME
	CPPFLAGS+= -DEPHOS_DEVICE_NAME=$(SYCL_DEVICE_NAME)
endif
ifdef SYCL_PLATFORM_NAME
	CPPFLAGS+= -DEPHOS_PLATFORM_NAME=$(SYCL_PLATFORM_NAME)
endif

ifdef SYCL_INCLUDE_PATH
	CPPFLAGS+= -I$(SYCL_INCLUDE_PATH)
endif
ifdef SYCL_LIBRARY_PATH
	LDDFLAGS+= -L$(SYCL_LIBRARY_PATH)
endif
LDDFLAGS+= -lComputeCpp
SYCLFLAGS=-sycl
ifdef SYCL_KERNEL_FORMAT
	SYCLFLAGS+= -sycl-target $(SYCL_KERNEL_FORMAT)
else
	SYCLFLAGS+= -sycl-target spir64
endif
SYCLFLAGS+= -DCL_OPENCL_TARGET_VERSION=120
CPPFLAGS+= -DCL_OPENCL_TARGET_VERSION=120

# sycl optimization
SYCL_WORK_GROUP_SIZE=512
CPPFLAGS+= -DEPHOS_COMPUTE_GROUP_SIZE=$(SYCL_WORK_GROUP_SIZE)
ifdef ENABLE_SYCL_MEMCPY
$(info Enabled OpenCL memcpy/memset intrinsics)
	SYCLFLAGS+= -no-serial-memop
endif

# other options
ifdef TESTCASE_LIMIT
$(info Limiting to $(TESTCASE_LIMIT) test cases)
	CPPFLAGS+= -DEPHOS_TESTCASE_LIMIT=$(TESTCASE_LIMIT)
endif
ifdef ENABLE_TESTDATA_GEN
$(info Enabled result data generation)
	CPPFLAGS+= -DEPHOS_TESTDATA_GEN
endif

all: benchmark checkdata

benchmark: euclidean_clustering.o framework.o ../common/main.o ../common/compute_tools.o
	$(CXX) $^ -o $@ $(LDDFLAGS)

%.o: %.cpp
	$(CXX) -c $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -o $@ $<

euclidean_clustering.o: euclidean_clustering.cpp kernel/kernel.sycl
	$(CXX) -c $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -include kernel/kernel.sycl -o $@ euclidean_clustering.cpp

kernel/kernel.sycl: euclidean_clustering.cpp
	compute++ $(SYCLFLAGS) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) $<
	mv euclidean_clustering.sycl kernel/kernel.sycl
	mv euclidean_clustering-sycl* kernel/

checkdata:
ifeq ($(wildcard ../../../data/ec_input.dat),)
	$(warning ec_input.dat not found. Did you forget to extract the test data?)
endif
ifeq ($(wildcard ../../../data/ec_output.dat),)
	$(warning ec_output.dat not found. Did you forget to extract the test data?)
endif

clean:
	rm -f Makefile.deps benchmark euclidean_clustering.o framework.o \
	../common/main.o ../common/compute_tools.o \
	kernel/kernel.sycl kernel/euclidean_clustering-sycl*

Makefile.deps:
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS)  -MM -MG \
	euclidean_clustering.cpp framework.cpp \
	../common/main.cpp ../common/compute_tools.cpp > Makefile.deps


