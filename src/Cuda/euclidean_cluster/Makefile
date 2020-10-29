.DEFAULT_GOAL := all
.PHONY: all clean checkdata

-include Makefile.deps

NXX=nvcc
ifdef DEBUG
	NXXFLAGS+= -g -G
else
	NXXFLAGS+= -O3
endif
CXXFLAGS+= -std=c++11
CPPFLAGS+= -I../../base/benchmark -I../../base/euclidean_cluster

# cuda specific options
CUDA_DEVICE_ARCH=compute_62
NXXFLAGS+= -arch $(CUDA_DEVICE_ARCH)
ifneq ($(CUDA_DEVICE_ARCH),compute_62)
$(info Compiling for CUDA device architecture $(CUDA_DEVICE_ARCH))
endif
CUDA_BLOCK_SIZE=512
NXXFLAGS+= -DEPHOS_KERNEL_BLOCK_SIZE=$(CUDA_BLOCK_SIZE)
ifneq ($(CUDA_BLOCK_SIZE),512)
$(info CUDA kernel block size set to $(CUDA_BLOCK_SIZE))
endif

# algorithm options
CLOUD_DISTANCES_PER_PACKET=8
ifneq ($(CLOUD_DISTANCES_PER_PACKET),8)
$(info Working with distance packets with $(CLOUD_DISTANCES_PER_PACKET) elements)
endif
CPPFLAGS+= -DEPHOS_CLOUD_DISTANCES_PER_PACKET=$(CLOUD_DISTANCES_PER_PACKET)

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

benchmark: euclidean_clustering.o euclidean_clustering_base.o main.o
	$(NXX) $^ -o $@ $(LDDFLAGS)

%.o: %.cu
	$(NXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(NXXFLAGS) -c -o $@ $<

%.o: ../../base/benchmark/common/%.cpp
	$(NXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(NXXFLAGS) -c -o $@ $<

%.o: ../../base/euclidean_cluster/common/%.cpp
	$(NXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(NXXFLAGS) -c -o $@ $<

checkdata:
ifeq ($(wildcard ../../../data/ec_input.dat),)
	$(warning ec_input.dat not found. Did you forget to extract the test data?)
endif
ifeq ($(wildcard ../../../data/ec_output.dat),)
	$(warning ec_output.dat not found. Did you forget to extract the test data?)
endif

clean:
	rm -f Makefile.deps benchmark euclidean_clustering.o euclidean_clustering_base.o \
	main.o

Makefile.deps:
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS)  -MM -MG \
	euclidean_clustering.cu ../../base/benchmark/common/main.cpp \
	../../base/euclidean_cluster/common/euclidean_clustering_base.cpp > Makefile.deps

