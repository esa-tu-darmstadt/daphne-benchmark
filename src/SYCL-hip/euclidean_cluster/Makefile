.DEFAULT_GOAL := all
.PHONY: all clean checkdata

-include Makefile.deps

CXXFLAGS=-O3
#CXXFLAGS=-g -fprofile-instr-generate -fcoverage-mapping
#LDDFLAGS=-fprofile-instr-generate
CXX=syclcc-clang
CPPFLAGS+= -I.

ifdef SYCL_DEVICE_TYPE
	CPPFLAGS+= -DEPHOS_DEVICE_TYPE=$(SYCL_DEVICE_TYPE)
endif

ifdef SYCL_INCLUDE_PATH
	CPPFLAGS+= -I$(SYCL_INCLUDE_PATH)
endif
ifdef SYCL_GPU_ARCH
	LDDFLAGS+= --hipsycl-gpu-arch=$(SYCL_GPU_ARCH)
	CXXFLAGS+= --hipsycl-gpu-arch=$(SYCL_GPU_ARCH)
else
	LDDFLAGS+= --hipsycl-gpu-arch=sm_60
	CXXFLAGS+= --hipsycl-gpu-arch=sm_60
endif



all: kernel checkdata

kernel: ../common/main.o kernel.o sycl/sycl_tools.o
	syclcc-clang $(LDDFLAGS) $^ -o $@

../common/main.o: ../common/main.cpp
	syclcc-clang -c $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -I../include $< -o $@

kernel.o: kernel.cpp
	syclcc-clang -c $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -I../include $< -o $@

sycl/sycl_tools.o: sycl/sycl_tools.cpp
	syclcc-clang -c $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) $< -o $@

checkdata:
ifeq ($(wildcard ../../../data/p2i_input.dat),)
	$(warning p2i_input.dat not found. Did you forget to extract the test data?)
endif
ifeq ($(wildcard ../../../data/p2i_output.dat),)
	$(warning p2i_output.dat not found. Did you forget to extract the test data?)
endif

clean:
	rm -f kernel kernel.o ../common/main.o sycl/sycl_tools.o Makefile.deps 

Makefile.deps:
	syclcc-clang $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -I../include -MM -MG ../common/main.cpp *.cpp sycl/sycl_tools.cpp > Makefile.deps

