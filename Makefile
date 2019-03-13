
.PHONY: clean

cpu:
	$(MAKE) -C src/CPU

cuda:
	$(MAKE) -C src/Cuda

opencl:
	$(MAKE) -C src/OpenCl

openmp:
	$(MAKE) -C src/OpenMP

all: cpu cuda opencl openmp

clean:
	@$(MAKE) -C src/CPU clean
	@$(MAKE) -C src/Cuda clean
	@$(MAKE) -C src/OpenCl clean
	@$(MAKE) -C src/OpenMP clean	
