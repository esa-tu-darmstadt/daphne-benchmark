* Preparation

  Extract the test case data files inside the data folder

  For points2image:
  * p2i_input.dat.gz
  * p2i_output.dat.gz

  For euclidean_cluster:
  * ec_input.dat.gz
  * ec_output.dat.gz

  For ndt_mapping:
  * ndt_input.dat.gz
  * ndt_output.dat.gz

  Example:
  $ gunzip --keep ec_input.dat.gz ec_output.dat.gz

  which yields:
  * ec_input.dat
  * ec_output.dat
  inside the data folder

* Compilation

  The compilation process requires a working hipSYCL installation.
  It makes use of the syclcc-clang tool from hipSYCL which must be in PATH.

  To compile all kernels in their default configuration:
  $ make

  The same result can be achieved in the kernel subfolder (points2image/eucldiean_cluster/ndt_mapping):
  $ make

  This sets the GPU Architecture to sm_60 which should work with the CUDA backend.
  In case of the ROCm backend or hardware limitations the GPU architecture can be set with the SYCL_GPU_ARCH variable:
  $ make SYCL_GPU_ARCH=sm_35

  The device type to use is configurable through the SYCL_DEVICE_TYPE variable.
  It can be set to GPU, CPU or HOST and defaults to GPU.
  Note that certain sycl implementations do not support all device types which generates errors at runtime.
  hipSYCL enables GPU calculations through CUDA or ROCm and does not support CPU or host devices at the time of writing.

  The Makefile variable SYCL_INCLUDE_PATH specifies the location of sycl headers.
  It can be set if the system can not find these files:
  $ make SYCL_INCLUDE_PATH=/random/path/include 

* Execute the benchmark

  In the kernel subfolder:
  $ ./kernel

  This will print information about the kernel runtime and unexpected deviations from the reference results
