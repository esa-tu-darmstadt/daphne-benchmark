.DEFAULT_GOAL := kernel

-include Makefile.deps

CFLAGS = -O3

kernel: ../common/main.o kernel.o 
	$(CXX) $^ -o $@

../common/main.o: ../common/main.cpp
	$(CXX) -c $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -I../include $< -o $@

kernel.o: kernel.cpp
	$(CXX) -c $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -I../include $< -o $@

.PHONY: clean
clean:
	rm -f kernel kernel.o ../common/main.o Makefile.deps

Makefile.deps:
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -I../include -MM ../common/main.cpp *.cpp > Makefile.deps

