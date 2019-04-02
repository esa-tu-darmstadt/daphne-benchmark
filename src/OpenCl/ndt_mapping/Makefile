.DEFAULT_GOAL := kernel
.PHONY: clean stringify

-include Makefile.deps

CPPFLAGS += -std=c++11

# debugging
DEBUG=0
ifeq ($(DEBUG),1)
	CPPFLAGS += -g -Wall -O0
else
	CPPFLAGS += -O3
endif
# work items per work group
LOCAL_SIZE=256 
CPPFLAGS += -DNUMWORKITEMS_PER_WORKGROUP=$(LOCAL_SIZE)

# opencl header and library
OPENCL_INCLUDE_PATH=
OPENCL_LIBRARY_PATH=
ifneq ($(OPENCL_INCLUDE_PATH),)
	CPPFLAGS += -I${OPENCL_INCLUDE_PATH}
endif
ifneq ($(OPENCL_LIBRARY_PATH),)
	LDDFLAGS += -L$(OPENCL_LIBRARY_PATH)
endif
LDDFLAGS += -lOpenCL

CPPFLAGS += -I./ocl/host -I./
CPPFLAGS += -DDOUBLE_FP

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

