.DEFAULT_GOAL := all
.PHONY: all clean checkdata kernel/kernel.h

-include Makefile.deps

ifdef DEBUG
	CXXFLAGS= -g
else
	CXXFLAGS= -O3
endif
CXXFLAGS+= -std=c++11
CPPFLAGS+= -I.. -I../../common/benchmark -I../../common/points2image

# opencl library
ifdef OPENCL_INCLUDE_PATH
$(info Custom OpenCL include path: $(OPENCL_INCLUDE_PATH))
	CPPFLAGS+= -I$(OPENCL_INCLUDE_PATH)
endif
ifdef OPENCL_LIBRARY_PATH
$(info Custom OpenCL library path: $(OPENCL_LIBRARY_PATH))
	LDDFLAGS+= -L$(OPENCL_LIBRARY_PATH)
endif
LDDFLAGS += -lOpenCL

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
ifdef ENABLE_TESTDATA_LEGACY
$(info Indicated legacy reference data)
	CPPFLAGS+= -DEPHOS_TESTDATA_LEGACY
endif
ifdef ENABLE_TESTDATA_GEN
$(info Enabled result data generation)
	CPPFLAGS+= -DEPHOS_TESTDATA_GEN
endif

all: benchmark checkdata

benchmark: main.o compute_tools.o points2image_base.o points2image.o
	$(CXX) $^ -o $@ $(LDDFLAGS)

%.o: %.cpp
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

%.o: ../../common/benchmark/%.cpp
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

%.o: ../../common/points2image/common/%.cpp
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

%.o: ../common/%.cpp
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
	rm -f benchmark points2image.o points2image_base.o kernel/kernel.h \
	main.o compute_tools.o Makefile.deps

Makefile.deps:
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -MM -MG \
	../../common/benchmark/main.cpp ../common/compute_tools.cpp \
	../../common/points2image/common/points2image_base.cpp \
	points2image.cpp > Makefile.deps

