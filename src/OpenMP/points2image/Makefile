.DEFAULT_GOAL := all
.PHONY: all clean checkdata kernel/kernel.h

-include Makefile.deps

ifdef DEBUG
	CXXFLAGS+= -g
else
	CXXFLAGS+= -O3
endif
CXXFLAGS+= -std=c++11
CPPFLAGS+= -I../../base/benchmark -I../../base/points2image
CPPFLAGS+= -fopenmp
LDDFLAGS+= -fopenmp

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
	$(CXX) $^ -o $@ $(LDDFLAGS)

%.o: %.cpp
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

%.o: ../../base/points2image/common/%.cpp
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

%.o: ../../base/benchmark/common/%.cpp
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

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
	points2image.cpp > Makefile.deps

