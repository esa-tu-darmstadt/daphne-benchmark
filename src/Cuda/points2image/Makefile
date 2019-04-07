.DEFAULT_GOAL := all
.PHONY: all clean checkdata

-include Makefile.deps

CXXFLAGS=-O3
CXXFLAGS+= -std c++11
CXX=nvcc
# cuda device capability
CUDA_DEVICE_ARCH=compute_62
CXXFLAGS+= -arch $(CUDA_DEVICE_ARCH)

all: kernel checkdata

kernel: ../common/main.o kernel.o
	$(CXX) $(CXXFLAGS) $^ -o $@

../common/main.o: ../common/main.cpp
	$(CXX) -c $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -I../include $< -o $@

kernel.o: kernel.cu
	$(CXX) -c $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -I../include $< -o $@

checkdata:
ifeq ($(wildcard ../../../data/p2i_input.dat),)
	$(warning p2i_input.dat not found. Did you forget to extract the test data?)
endif
ifeq ($(wildcard ../../../data/p2i_output.dat),)
	$(warning p2i_output.dat not found. Did you forget to extract the test data?)
endif

clean:
	rm -f kernel kernel.o ../common/main.o Makefile.deps

Makefile.deps:
	g++ $(CFLAGS) $(CPPFLAGS)  -I../include -MM ../common/main.cpp *.cu > Makefile.deps

