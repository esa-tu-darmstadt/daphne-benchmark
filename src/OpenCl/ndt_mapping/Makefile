.DEFAULT_GOAL := kernel

-include Makefile.deps

# debugging
DEBUG=0
# work items per work group
LOCAL_SIZE=256 
# optional paths
OPENCL_INCLUDE_PATH=
OPENCL_LIBRARY_PATH=

# preprocessor, compiler and linker flags
CPPFLAGS += -std=c++11
ifeq ($(DEBUG),1)
	CPPFLAGS += -g -Wall -O0
else
	CPPFLAGS += -O3
endif
ifneq ($(OPENCL_INCLUDE_PATH),)
	CPPFLAGS += -I${OPENCL_INCLUDE_PATH}
endif
CPPFLAGS += -I./ocl/host -I./
CPPFLAGS += -DNUMWORKITEMS_PER_WORKGROUP=${LOCAL_SIZE}
CPPFLAGS += -DDOUBLE_FP
ifneq ($(OPENCL_LIBRARY_PATH),)
	LDDFLAGS += -L$(OPENCL_LIBRARY_PATH)
endif
LDDFLAGS += -lOpenCL

stringify:
	./stringify_ocl_krnls.sh

kernel: ../common/main.o kernel.o 
	$(CXX) $^ -o $@ $(LDDFLAGS)

../common/main.o: ../common/main.cpp
	$(CXX) -c $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -I../include $< -o $@

kernel.o: kernel.cpp stringify
	$(CXX) -c $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -I../include $< -o $@

.PHONY: clean
clean:
	rm -f kernel kernel.o ../common/main.o Makefile.deps

Makefile.deps:
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -I../include -MM ../common/main.cpp *.cpp > Makefile.deps

