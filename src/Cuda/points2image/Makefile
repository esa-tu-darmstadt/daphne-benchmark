.DEFAULT_GOAL := kernel


CXXFLAGS=-std c++11 -O3 -arch compute_62 
CXX=nvcc

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

