.DEFAULT_GOAL := all
.PHONY: all clean checkdata kernel/kernel.h

-include Makefile.deps

ifdef DEBUG
	CXXFLAGS+= -g
else
	CXXFLAGS+= -O3
endif
CXXFLAGS+= -std=c++11
CPPFLAGS+= -I./ -I../


OPENCL_DISTANCE_PACKETS_PER_WORK_ITEM=1
OPENCL_DISTANCES_PER_PACKET=8
ifneq ($(OPENCL_DISTANCES_PER_PACKET),8)
$(info Working with distance packets with $(OPENCL_DISTANCES_PER_PACKET) elements)
endif
ifneq ($(OPENCL_DISTANCE_PACKETS_PER_WORK_ITEM),1)
$(info Processing $(OPENCL_DISTANCE_PACKETS_PER_WORK_ITEM) packets with each work item)
endif
CPPFLAGS+= -DEPHOS_KERNEL_DISTANCE_PACKETS_PER_ITEM=$(OPENCL_DISTANCE_PACKETS_PER_WORK_ITEM)
CPPFLAGS+= -DEPHOS_KERNEL_DISTANCES_PER_PACKET=$(OPENCL_DISTANCES_PER_PACKET)

# other options
ifdef TESTCASE_LIMIT
$(info Limiting to $(TESTCASE_LIMIT) test cases)
	CPPFLAGS+= -DTESTCASE_LIMIT=$(TESTCASE_LIMIT)
endif
ifdef ENABLE_DATAGEN
$(info Enabled result data generation)
	CPPFLAGS+= -DEPHOS_DATAGEN
endif

all: benchmark checkdata

benchmark: euclidean_clustering.o framework.o ../common/main.o
	$(CXX) $^ -o $@ $(LDDFLAGS)

%.o: %.cpp
	$(CXX) -c $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -o $@ $<

checkdata:
ifeq ($(wildcard ../../../data/ec_input.dat),)
	$(warning ec_input.dat not found. Did you forget to extract the test data?)
endif
ifeq ($(wildcard ../../../data/ec_output.dat),)
	$(warning ec_output.dat not found. Did you forget to extract the test data?)
endif

clean:
	rm -f Makefile.deps benchmark euclidean_clustering.o framework.o \
	../common/main.o

Makefile.deps:
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS)  -MM -MG \
	../common/main.cpp euclidean_clustering.cpp framework.cpp > Makefile.deps


