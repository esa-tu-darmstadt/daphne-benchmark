.DEFAULT_GOAL := all
.PHONY: all clean kernel/kernel.h checkdata

-include Makefile.deps

ifdef DEBUG
	CXXFLAGS+= -g
else
	CXXFLAGS=-O3
endif
CXXFLAGS+= -std=c++11
CPPFLAGS+= -I./ -I../
CPPFLAGS+= -fopenmp
LDDFLAGS+= -fopenmp


# other options
ifdef TESTCASE_LIMIT
$(info Testcase limit set to $(TESTCASE_LIMIT))
	CPPFLAGS+= -DEPHOS_TESTCASE_LIMIT=$(TESTCASE_LIMIT)
endif
ifdef ENABLE_DATAGEN
$(info Enabled result data generation)
	CPPFLAGS+= -DEPHOS_DATAGEN
endif
all: benchmark checkdata

benchmark: ndt_mapping.o framework.o ../common/main.o
	$(CXX) $^ -o $@ $(LDDFLAGS)

%.o: %.cpp
	$(CXX) -c $(CLFAGS) $(CPPFLAGS) $(CXXFLAGS) -o $@ $<

checkdata:
ifeq ($(wildcard ../../../data/ndt_input.dat),)
	$(warning ndt_input.dat not found. Did you forget to extract the test data?)
endif
ifeq ($(wildcard ../../../data/ndt_output.dat),)
	$(warning ndt_output.dat not found. Did you forget to extract the test data?)
endif

clean:
	rm -f Makefile.deps benchmark ndt_mapping.o ../common/main.o framework.o \
	../common/main.o

Makefile.deps:
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -MM -MG \
	../common/main.cpp ndt_mapping.cpp framework.cpp > Makefile.deps
