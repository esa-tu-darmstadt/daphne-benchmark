.DEFAULT_GOAL := all
.PHONY: all clean checkdata

-include Makefile.deps

CXXFLAGS=-O3
CXXFLAGS+= -std=c++11

all: kernel checkdata

kernel: ../common/main.o kernel.o 
	$(CXX) $^ -o $@

../common/main.o: ../common/main.cpp
	$(CXX) -c $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -I../include $< -o $@

kernel.o: kernel.cpp
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
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -I../include -MM ../common/main.cpp *.cpp > Makefile.deps

