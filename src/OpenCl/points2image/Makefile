.DEFAULT_GOAL := kernel
.PHONY: clean stringify

-include Makefile.deps

CPPFLAGS = -std=c++11

DEBUG=0
ifeq ($(DEBUG),1)
	CPPFLAGS += -g -Wall
else
	CPPFLAGS +=  -O3
endif

# opencl library
OPENCL_INCLUDE_PATH=
OPENCL_LIBRARY_PATH=
ifneq ($(OPENCL_INCLUDE_PATH),)
	CPPFLAGS += -I$(OPENCL_INCLUDE_PATH)
endif
ifneq ($(OPENCL_LIBRARY_PATH),)
	LDDFLAGS += -L$(OPENCL_LIBRARY_PATH)
endif
LDDFLAGS += -lOpenCL
CPPFLAGS += -I./ocl/host -I./

# opencl platform
OPENCL_PLATFORM_ID=
OPENCL_DEVICE_ID=
OPENCL_DEVICE_TYPE=
ifneq ($(OPENCL_PLATFORM_ID),)
	CPPFLAGS += -DEPHOS_PLATFORM_HINT=$(OPENCL_PLATFORM_ID)
endif
ifneq ($(OPENCL_DEVICE_ID),)
	CPPFLAGS += -DEPHOS_DEVICE_HINT=$(OPENCL_DEVICE_ID)
endif
ifneq ($(OPENCL_DEVICE_TYPE),)
	CPPFLAGS += -DEPHOS_DEVICE_TYPE=$(OPENCL_DEVICE_TYPE)
endif

stringify:
	./stringify_ocl_krnls.sh

kernel: ../common/main.o kernel.o
	$(CXX) $^ -o $@ $(LDDFLAGS)

../common/main.o: ../common/main.cpp
	$(CXX) -c $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -I../include $< -o $@

kernel.o: kernel.cpp stringify
	$(CXX) -c $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -I../include $< -o $@

clean:
	rm -f kernel kernel.o ../common/main.o Makefile.deps ocl/device/stringify_tmp

Makefile.deps:
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -I../include -MM ../common/main.cpp *.cpp > Makefile.deps

