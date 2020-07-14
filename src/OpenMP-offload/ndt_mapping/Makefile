.DEFAULT_GOAL := all
.PHONY: all clean checkdata

-include Makefile.deps

CXXFLAGS=-O3
#CXXFLAGS=-g
CXXFLAGS+= -std=c++11 -fopenmp
CXXFLAGS+= -lm -foffload=-lm

# device hints
EPHOS_TARGET_DEVICE_ID=
EPHOS_HOST_DEVICE_ID=
ifneq ($(OPENMP_TARGET_DEVICE_ID),)
        CPPFLAGS+= -DEPHOS_TARGET_DEVICE_ID=$(OPENMP_TARGET_DEVICE_ID)
endif
ifneq ($(OPENMP_HOST_DEVICE_ID),)
        CPPFLAGS+= -DEPHOS_HOST_DEVICE_ID=$(OPENMP_HOST_DEVICE_ID)
endif


all: kernel checkdata

kernel: ../common/main.o kernel.o 
	$(CXX) $(CXXFLAGS) $^ -o $@

../common/main.o: ../common/main.cpp
	$(CXX) -c $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -I../include $< -o $@

kernel.o: kernel.cpp
	$(CXX) -c $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -I../include $< -o $@

checkdata:
ifeq ($(wildcard ../../../data/ndt_input.dat),)
	$(warning ndt_input.dat not found. Did you forget to extract the test data?)
endif
ifeq ($(wildcard ../../../data/ndt_output.dat),)
	$(warning ndt_output.dat not found. Did you forget to extract the test data?)
endif

clean:
	rm -f kernel kernel.o ../common/main.o Makefile.deps

Makefile.deps:
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -I../include -MM ../common/main.cpp *.cpp > Makefile.deps

