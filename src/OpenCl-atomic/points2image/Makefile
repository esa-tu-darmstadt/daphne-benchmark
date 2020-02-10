.DEFAULT_GOAL := all
.PHONY: all clean checkdata kernel/kernel.cpp

CXXFLAGS=-O3
CXXFLAGS+= -std=c++11

# opencl library
ifdef OPENCL_INCLUDE_PATH
$(info Custom OpenCL include path: $(OPENCL_INCLUDE_PATH))
	CPPFLAGS+= -I$(OPENCL_INCLUDE_PATH)
endif
ifdef OPENCL_LIBRARY_PATH
$(info Custom OpenCL library path: $(OPENCL_LIBRARY_PATH))
	LDDFLAGS+= -L$(OPENCL_LIBRARY_PATH)
endif

# opencl platform
ifdef OPENCL_PLATFORM_ID
$(info OpenCL platform: $(OPENCL_PLATFORM_ID))
	CPPFLAGS+= -DEPHOS_PLATFORM_HINT=$(OPENCL_PLATFORM_ID)
endif
ifdef OPENCL_DEVICE_ID
$(info OpenCL device: $(OPENCL_DEVICE_ID))
	CPPFLAGS+= -DEPHOS_DEVICE_HINT=$(OPENCL_DEVICE_ID)
endif
ifdef OPENCL_DEVICE_TYPE
$(info OpenCL device type: $(OPENCL_DEVICE_TYPE))
	CPPFLAGS+= -DEPHOS_DEVICE_TYPE=$(OPENCL_DEVICE_TYPE)
endif

# opencl optimization
ifdef ENABLE_OPENCL_ZERO_COPY
$(info Enabled OpenCL zero copy memory)
	CPPFLAGS+= -DEPHOS_ZERO_COPY=1
endif
ifdef ENABLE_OPENCL_PINNED_MEMORY
$(info Enabled OpenCL page-locked memory)
	CPPFLAGS+= -DEPHOS_PINNED_MEMORY=1
endif
ifndef DISABLE_OPENCL_ATOMICS
	CPPFLAGS+= -DEPHOS_KERNEL_ATOMICS=1
else
$(info Enabled OpenCL atomics)
endif
ifndef DISABLE_OPENCL_LOCAL_ATOMICS
	CPPFLAGS+= -DEPHOS_KERNEL_LOCAL_ATOMICS=1
else
$(info Disabled local OpenCL atomics)
endif
ifdef OPENCL_TRANSFORMS_PER_WORK_ITEM
$(info Transformations per work item: $(OPENCL_TRANSFORMS_PER_WORK_ITEM))
	CPPFLAGS+= -DEPHOS_KERNEL_TRANSFORMS_PER_ITEM=$(OPENCL_TRANSFORMS_PER_WORK_ITEM)
endif
OPENCL_WORK_GROUP_SIZE=512
CPPFLAGS+= -DEPHOS_KERNEL_WORK_GROUP_SIZE=$(OPENCL_WORK_GROUP_SIZE)

# other settings
ifdef TESTCASE_LIMIT
	CPPFLAGS+= -DEPHOS_TESTCASE_LIMIT=$(TESTCASE_LIMIT)
endif
# whether to use a sparse reference image representation for improved read speeds
ifdef TESTCASE_SPARSE
	CPPFLAGS+= -DEPHOS_TESTCASE_SPARSE=1
endif

CPPFLAGS+= -I./ -I../
LDDFLAGS += -lOpenCL

-include Makefile.deps

all: points2image checkdata

points2image: ../common/main.o ../common/compute_tools.o points2image.o framework.o
	$(CXX) $^ -o $@ $(LDDFLAGS)

%.o: %.cpp
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

kernel/kernel.h:
	cd kernel && ./stringify_kernels.sh

checkdata:
ifeq ($(wildcard ../../../data/p2i_input.dat),)
	$(warning p2i_input.dat not found. Did you forget to extract the test data?)
endif
ifeq ($(wildcard ../../../data/p2i_output.dat),)
	$(warning p2i_output.dat not found. Did you forget to extract the test data?)
endif

clean:
	rm -f points2image points2image.o framework.o kernel/kernel.h \
	../common/main.o ../common/compute_tools.o Makefile.deps

Makefile.deps:
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -MM -MG \
	../common/main.cpp ../common/compute_tools.cpp points2image.cpp framework.cpp > Makefile.deps

