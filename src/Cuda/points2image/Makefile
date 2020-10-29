.DEFAULT_GOAL := all
.PHONY: all clean checkdata kernel/kernel.h

-include Makefile.deps

NXX=nvcc
ifdef DEBUG
	NXXFLAGS= -g -G
else
	NXXFLAGS= -O3
endif
CXXFLAGS+= -std=c++11
CPPFLAGS+= -I../../base/benchmark -I../../base/points2image

# cuda specific options
CUDA_DEVICE_ARCH=compute_62
NXXFLAGS+= -arch $(CUDA_DEVICE_ARCH)

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

benchmark: main.o points2image_base.o points2image.o
	$(NXX) $^ -o $@ $(LDDFLAGS)

%.o: %.cu
	$(NXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(NXXFLAGS) -c -o $@ $<

%.o: ../../base/points2image/common/%.cpp
	$(NXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(NXXFLAGS) -c -o $@ $<

%.o: ../../base/benchmark/common/%.cpp
	$(NXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(NXXFLAGS) -c -o $@ $<

checkdata:
ifeq ($(wildcard ../../../data/p2i_input.dat),)
	$(warning p2i_input.dat not found. Did you forget to extract the test data?)
endif
ifeq ($(wildcard ../../../data/p2i_output.dat),)
	$(warning p2i_output.dat not found. Did you forget to extract the test data?)
endif

clean:
	rm -f benchmark points2image.o points2image_base.o main.o Makefile.deps

Makefile.deps:
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -MM -MG \
	../../base/benchmark/common/main.cpp ../../base/points2image/common/points2image_base.cpp \
	points2image.cu > Makefile.deps

