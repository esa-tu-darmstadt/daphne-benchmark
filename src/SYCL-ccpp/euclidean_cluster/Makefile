.DEFAULT_GOAL := all
.PHONY: all clean checkdata

-include Makefile.deps

CXXFLAGS=-O3
#CXXFLAGS=-g
CXXFLAGS+= -std=c++11

SYCL_DEVICE_TYPE=GPU
CPPFLAGS+= -DEPHOS_DEVICE_TYPE=$(SYCL_DEVICE_TYPE)
CPPFLAGS+= -I.

SYCL_INCLUDE_PATH=
SYCL_LIBRARY_PATH=
ifneq ($(SYCL_INCLUDE_PATH),)
	CPPFLAGS+= -I$(SYCL_INCLUDE_PATH)
endif
ifneq ($(SYCL_LIBRARY_PATH),)
	LDDFLAGS+= -L$(SYCL_LIBRARY_PATH)
endif
LDDFLAGS+= -lComputeCpp
SYCLFLAGS=-sycl
SYCLFLAGS+= -sycl-target ptx64

all: kernel checkdata

kernel: ../common/main.o kernel.o sycl/sycl_tools.o
	$(CXX) $(LDDFLAGS) $^ -o $@

../common/main.o: ../common/main.cpp
	$(CXX) -c $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -I../include $< -o $@

kernel.o: kernel.cpp sycl/kernel.sycl
	$(CXX) -c $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -include sycl/kernel.sycl -I../include $< -o $@

sycl/kernel.sycl: kernel.cpp
	compute++ $(SYCLFLAGS) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -I../include -I/usr/include/c++/7 -I/usr/include/aarch64-linux-gnu/c++/7/bits kernel.cpp
	mv kernel.sycl sycl/
	mv kernel-sycl* sycl/

sycl/sycl_tools.o: sycl/sycl_tools.cpp
	$(CXX) -c $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) $< -o $@

checkdata:
ifeq ($(wildcard ../../../data/ec_input.dat),)
	$(warning ec_input.dat not found. Did you forget to extract the test data?)
endif
ifeq ($(wildcard ../../../data/ec_output.dat),)
	$(warning ec_output.dat not found. Did you forget to extract the test data?)
endif

clean:
	rm -f kernel kernel.o ../common/main.o \
	sycl/sycl_tools.o sycl/kernel.sycl sycl/kernel-sycl* \
	Makefile.deps

Makefile.deps:
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -I../include -MM -MG \
	../common/main.cpp *.cpp sycl/sycl_tools.cpp > Makefile.deps

