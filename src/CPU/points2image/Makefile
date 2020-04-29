.DEFAULT_GOAL := all
.PHONY: all clean checkdata kernel/kernel.h

-include Makefile.deps

ifdef DEBUG
	CXXFLAGS= -g
else
	CXXFLAGS= -O3
endif
CXXFLAGS+= -std=c++11
CPPFLAGS+= -I./ -I../

# other settings
ifdef TESTCASE_LIMIT
	CPPFLAGS+= -DEPHOS_TESTCASE_LIMIT=$(TESTCASE_LIMIT)
endif
# whether to use a sparse reference image representation for improved read speeds
ifdef TESTCASE_SPARSE
$(info Indicated sparse reference data)
	CPPFLAGS+= -DEPHOS_TESTCASE_SPARSE=1
endif
ifdef ENABLE_DATAGEN
$(info Enabled result data generation)
	CPPFLAGS+= -DEPHOS_DATAGEN
endif

all: benchmark checkdata

benchmark: ../common/main.o points2image.o framework.o
	$(CXX) $^ -o $@ $(LDDFLAGS)

%.o: %.cpp
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

kernel/kernel.h:
	cd kernel && ./stringify_kernels.sh

checkdata:
ifeq ($(wildcard ../../../data/p2i_input.dat),)
	$(warning p2i_input.dat not found. Did you forget to extract the test data?)
endif
ifeq ($(wildcard ../../../data/p2i_output.dat),)
	$(warning p2i_output.dat not found. Did you forget to extract the test data?)
endif

clean:
	rm -f benchmark points2image.o framework.o kernel/kernel.h \
	../common/main.o  Makefile.deps

Makefile.deps:
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -MM -MG \
	../common/main.cpp points2image.cpp framework.cpp > Makefile.deps

