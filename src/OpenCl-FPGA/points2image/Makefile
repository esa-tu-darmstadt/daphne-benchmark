REPORT := none
TARGET := hw
PLATFORM := zcu102
BUILD_DIR := ./build/$(PLATFORM)_$(TARGET)

CXX := aarch64-linux-gnu-g++
XOCC := xocc

# Points to Utility Directory
#COMMON_REPO = ../../../
COMMON_REPO = /home/lvs/ESA/SDSoC_Examples/
ABS_COMMON_REPO = $(shell readlink -f $(COMMON_REPO))

# Include Libraries
include $(ABS_COMMON_REPO)/libs/xcl2/xcl2.mk

# -----------------------------------------------------------------
# If emulation, then change folders for input/output

ifeq ($(TARGET),sw_emu)
        CXXFLAGS += -DSW_EMU
else
        CXXFLAGS +=
endif
# -----------------------------------------------------------------
# Printing information
PRINTINFO=0

ifeq ($(PRINTINFO),1)
        CXXFLAGS += -DPRINTINFO
else
        CXXFLAGS +=
endif

# -----------------------------------------------------------------
OPENCL=0

ifeq ($(OPENCL),1)
	CXXFLAGS += -DXILINX_OPENCL_EPHOS -I./ocl/host/
else
	CXXFLAGS +=
endif
# -----------------------------------------------------------------

# Host compiler global settings
#CXXFLAGS += -I $(XILINX_SDX)/runtime/include/1_2/ -I/$(XILINX_SDX)/Vivado_HLS/include/ -O2 -g -Wall -fmessage-length=0 -std=c++14
#CXXFLAGS += -I $(XILINX_SDX)/runtime/include/1_2/ \
#            -I/$(XILINX_SDX)/Vivado_HLS/include/ \
#            -I./ \
#            -I../include/ \
#            -O2 -g -Wall -fmessage-length=0 -std=c++14

CXXFLAGS += -I $(XILINX_SDX)/runtime/include/1_2/ \
            -I/$(XILINX_SDX)/Vivado_HLS/include/ \
            -I./ \
            -I../include/ \
            -O1 -g -Wall -fmessage-length=0 -std=c++14

LDFLAGS += -lxilinxopencl -lpthread -lrt -ldl -lcrypt -lstdc++ -L$(XILINX_SDX)/runtime/lib/aarch64 $(xcl2_CXXFLAGS)

# Kernel compiler global settings
#CLFLAGS = -t hw --platform $(PLATFORM) --save-temps  --clkid 0
CLFLAGS = -t $(TARGET) --platform $(PLATFORM) --save-temps  --clkid 0

ifneq ($(REPORT), none)
CLFLAGS += --report $(REPORT)
endif

EXECUTABLE = pointcloud2_to_image

#BINARY_CONTAINERS += $(BUILD_DIR)/vadd.xclbin
#BINARY_CONTAINER_1_OBJS += $(BUILD_DIR)/vadd.xo
#ALL_KERNEL_OBJS += $(BUILD_DIR)/vadd.xo

BINARY_CONTAINERS += $(BUILD_DIR)/$(EXECUTABLE).xclbin
BINARY_CONTAINER_1_OBJS += $(BUILD_DIR)/$(EXECUTABLE).xo
ALL_KERNEL_OBJS += $(BUILD_DIR)/$(EXECUTABLE).xo

ALL_MESSAGE_FILES = $(subst .xo,.mdb,$(ALL_KERNEL_OBJS)) $(subst .xclbin,.mdb,$(BINARY_CONTAINERS))

HOST_SRCS = kernel.cpp \
            ../common/main.cpp \
            $(xcl2_SRCS)

#EXECUTABLE = vadd

CP = cp -rf

.PHONY: all clean cleanall docs
all: $(BUILD_DIR)/$(EXECUTABLE) $(BINARY_CONTAINERS)
	- $(CP) $(BUILD_DIR)/$(EXECUTABLE) $(BUILD_DIR)/sd_card/

# Building kernel
$(BUILD_DIR)/$(EXECUTABLE).xo: ./ocl/device/$(EXECUTABLE).cl
	mkdir -p $(BUILD_DIR)
	-@$(RM) $@
	@echo 'Building Target: $@'
	$(XOCC) $(CLFLAGS) -c -k $(EXECUTABLE) --messageDb $(subst .xo,.mdb,$@) -I'$(<D)' --xp misc:solution_name=_xocc_compile_$(EXECUTABLE) -o'$@' '$<'
	@echo 'SDx Completed Building Target: $@'

$(BUILD_DIR)/$(EXECUTABLE).xclbin: $(BINARY_CONTAINER_1_OBJS)
	$(XOCC) $(CLFLAGS) -l --nk $(EXECUTABLE):1 --messageDb $(subst .xclbin,.mdb,$@) --xp misc:solution_name=_xocc_link_$(EXECUTABLE) -o'$@' $(+) --sys_config ocl

# Building Host
$(BUILD_DIR)/$(EXECUTABLE): $(HOST_SRCS)
	mkdir -p $(BUILD_DIR)
	@echo 'Building Target: $@'
	$(CXX) -o '$@' $(HOST_SRCS) $(CXXFLAGS) $(LDFLAGS)
	@echo 'SDx Completed Building Target: $@'

# Cleaning stuff
RM = rm -f
RMDIR = rm -rf
clean:
	-$(RM) $(ALL_MESSAGE_FILES) $(EXECUTABLE)
	-$(RMDIR) embedded_root _xocc_* iprepo _sds
	-$(RM) *.xml *.tcl *.bat image.ub
	-$(RM) init.sh *.bit *.hdf $(PLATFORM)_* *.dat
	-$(RM) platform_desc.txt *.xtxt *.ko *.hpfm *.so
	-$(RMDIR) .Xil

cleanall: clean
	-$(RM) $(BINARY_CONTAINERS) $(ALL_KERNEL_OBJS) $(EXECUTABLE)
	-$(RMDIR) $(BUILD_DIR) ./build

docs: README.md

README.md: description.json
	$(ABS_COMMON_REPO)/ocl/common/readme_gen/readme_gen.py description.json

