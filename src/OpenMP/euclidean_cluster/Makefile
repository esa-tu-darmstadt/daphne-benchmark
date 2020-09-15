.DEFAULT_GOAL := all
.PHONY: all clean checkdata kernel/kernel.h

-include Makefile.deps

ifdef DEBUG
	CXXFLAGS+= -g
else
	CXXFLAGS+= -O3
endif
CXXFLAGS+= -std=c++11
CPPFLAGS+= -I../../base/benchmark -I../../base/euclidean_cluster
CPPFLAGS+= -fopenmp
LDDFLAGS+= -fopenmp

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
	$(CXX) $^ -o $@ $(LDDFLAGS)

%.o: %.cpp
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

%.o: ../../base/benchmark/common/%.cpp
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

%.o: ../../base/euclidean_cluster/common/%.cpp
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

checkdata:
ifeq ($(wildcard ../../../data/ec_input.dat),)
	$(warning ec_input.dat not found. Did you forget to extract the test data?)
endif
ifeq ($(wildcard ../../../data/ec_output.dat),)
	$(warning ec_output.dat not found. Did you forget to extract the test data?)
endif

clean:
	rm -f Makefile.deps benchmark euclidean_clustering.o euclidean_clustering_base.o \
	main.o benchmark

Makefile.deps:
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS)  -MM -MG \
	euclidean_clustering.cpp ../../base/benchmark/common/main.cpp \
	../../base/euclidean_cluster/common/euclidean_clustering_base.cpp > Makefile.deps


