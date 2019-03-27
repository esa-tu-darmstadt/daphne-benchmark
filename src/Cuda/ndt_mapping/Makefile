.DEFAULT_GOAL := kernel


#CXXFLAGS=-std c++11 -g -O3 -arch=sm_72 
CXXFLAGS=-std c++11 -g -O3 -arch=sm_60
CXX=nvcc
#CXXFLAGS=-std=gnu++11 -O3


-include Makefile.deps

kernel: ../common/main.o kernel.o
	$(CXX) $(CXXFLAGS) $^ -o $@

../common/main.o: ../common/main.cpp
	$(CXX) -c $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -I../include $< -o $@

kernel.o: kernel.cu
	$(CXX) -c $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -I../include $< -o $@

.PHONY: clean
clean:
	rm -f kernel kernel.o ../common/main.o Makefile.deps

Makefile.deps:
	g++ $(CFLAGS) $(CPPFLAGS)  -I../include -MM ../common/main.cpp *.cu > Makefile.deps

