.DEFAULT_GOAL := all
.PHONY: all clean checkdata kernel/kernel.h

-include Makefile.deps

ifdef DEBUG
	CXXFLAGS= -g
else
	CXXFLAGS= -O3
endif
CXXFLAGS+= -std=c++11
CPPFLAGS+= -I./ -I../

# sycl platform flags
ifdef SYCL_DEVICE_TYPE
$(info Selecting SYCL device type $(SYCL_DEVICE_TYPE))
	CPPFLAGS+= -DEPHOS_DEVICE_TYPE=$(SYCL_DEVICE_TYPE)
endif
ifdef SYCL_DEVICE_NAME
$(info Selecting SYCL device $(SYCL_DEVICE_NAME))
	CPPFLAGS+= -DEPHOS_DEVICE_NAME=$(SYCL_DEVICE_NAME)
endif
ifdef SYCL_PLATFORM_NAME
$(info Selecting SYCL platform $(SYCL_PLATFORM_NAME))
	CPPFLAGS+= -DEPHOS_PLATFORM_NAME=$(SYCL_PLATFORM_NAME)
endif
SYCL_WORK_GROUP_SIZE=512
CPPFLAGS+= -DEPHOS_COMPUTE_GROUP_SIZE=$(SYCL_WORK_GROUP_SIZE)

ifdef SYCL_INCLUDE_PATH
	CPPFLAGS+= -I$(SYCL_INCLUDE_PATH)
endif
ifdef SYCL_LIBRARY_PATH
	LDDFLAGS+= -L$(SYCL_LIBRARY_PATH)
endif
LDDFLAGS+= -lComputeCpp

SYCLFLAGS= -sycl
ifdef SYCL_KERNEL_FORMAT
$(info Compiling to kernel format $(SYCL_KERNEL_FORMAT))
	SYCLFLAGS+= -sycl-target $(SYCL_KERNEL_FORMAT)
else
	SYCLFLAGS+= -sycl-target spir64
endif
CPPFLAGS+= -DCL_TARGET_OPENCL_VERSION=120
SYCLFLAGS+= -DCL_TARGET_OPENCL_VERSION=120
ifdef ENABLE_SYCL_MEMCPY
$(info Enabled SYCL memcpy operations)
	SYCLFLAGS+= -no-serial-memop
endif
ifdef CUDA_GPU_ARCH
$(info Using $(CUDA_GPU_ARCH) GPU architecture)
	# ignored during compilation
	SYCLFLAGS+= --cuda-gpu-arch=$(CUDA_GPU_ARCH)
endif


# other settings
ifdef TESTCASE_LIMIT
$(info Limiting to $(TESTCASE_LIMIT) test cases)
	CPPFLAGS+= -DEPHOS_TESTCASE_LIMIT=$(TESTCASE_LIMIT)
endif
ifdef ENABLE_LEGACY_TESTDATA
$(info Indicated legacy reference data)
	CPPFLAGS+= -DEPHOS_TESTDATA_LEGACY
endif
ifdef ENABLE_TESTDATA_GEN
$(info Enabled result data generation)
	CPPFLAGS+= -DEPHOS_TESTDATA_GEN
endif

all: benchmark checkdata

benchmark: ../common/main.o points2image.o framework.o ../common/compute_tools.o
	$(CXX) $^ -o $@ $(LDDFLAGS)

%.o: %.cpp
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

points2image.o: points2image.cpp kernel/kernel.sycl
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -include kernel/kernel.sycl -c -o $@ $<

kernel/kernel.h:
	cd kernel && ./stringify_kernels.sh

kernel/kernel.sycl: points2image.cpp
	compute++ $(SYCLFLAGS) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) $<
	mv points2image.sycl kernel/kernel.sycl
	mv points2image-sycl* kernel/

checkdata:
ifeq ($(wildcard ../../../data/p2i_input.dat),)
	$(warning p2i_input.dat not found. Did you forget to extract the test data?)
endif
ifeq ($(wildcard ../../../data/p2i_output.dat),)
	$(warning p2i_output.dat not found. Did you forget to extract the test data?)
endif

clean:
	rm -f benchmark points2image.o framework.o kernel/kernel.h \
	../common/main.o  Makefile.deps
	rm kernel/kernel.sycl
	rm kernel/points2image-sycl*

Makefile.deps:
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -MM -MG \
	../common/main.cpp points2image.cpp framework.cpp > Makefile.deps

