.DEFAULT_GOAL := all
.PHONY: all clean kernel/kernel.h checkdata

-include Makefile.deps

NXX=nvcc
ifdef DEBUG
	CXXFLAGS+= -g -G
else
	CXXFLAGS+= -O3
endif
CXXFLAGS+= -std=c++11
CPPFLAGS+= -I../../base/benchmark -I../../base/ndt_mapping

# cuda compiler options
CUDA_DEVICE_ARCH=compute_72
NXXFLAGS+= -arch $(CUDA_DEVICE_ARCH)
ifneq ($(CUDA_DEVICE_ARCH),compute_72)
$(info Compiling for CUDA device architecture $(CUDA_DEVICE_ARCH))
endif
CUDA_BLOCK_SIZE=512
NXXFLAGS+= -DEPHOS_KERNEL_BLOCK_SIZE=$(CUDA_BLOCK_SIZE)
ifneq ($(CUDA_BLOCK_SIZE),512)
$(info CUDA kernel block size set to $(CUDA_BLOCK_SIZE))
endif

# other options
ifdef TESTCASE_LIMIT
$(info Testcase limit set to $(TESTCASE_LIMIT))
	CPPFLAGS+= -DEPHOS_TESTCASE_LIMIT=$(TESTCASE_LIMIT)
endif
ifdef ENABLE_DATAGEN
$(info Enabled result data generation)
	CPPFLAGS+= -DEPHOS_DATAGEN
endif
all: benchmark checkdata

benchmark: ndt_mapping.o ndt_mapping_base.o framework.o main.o
	$(NXX) $^ -o $@ $(LDDFLAGS)

%.o: %.cu
	$(NXX) $(CLFAGS) $(CPPFLAGS) $(CXXFLAGS) $(NXXFLAGS) -c -o $@ $<

%.o: ../../base/benchmark/common/%.cpp
	$(NXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(NXXFLAGS) -c -o $@ $<

%.o: ../../base/ndt_mapping/common/%.cpp
	$(NXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(NXXFLAGS) -c -o $@ $<

checkdata:
ifeq ($(wildcard ../../../data/ndt_input.dat),)
	$(warning ndt_input.dat not found. Did you forget to extract the test data?)
endif
ifeq ($(wildcard ../../../data/ndt_output.dat),)
	$(warning ndt_output.dat not found. Did you forget to extract the test data?)
endif

clean:
	rm -f Makefile.deps benchmark ndt_mapping.o ndt_mapping_base.o main.o

Makefile.deps:
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -MM -MG \
	../../base/benchmark/common/main.cpp ../../base/ndt_mapping/common/ndt_mapping_base.cpp \
	ndt_mapping.cu > Makefile.deps
