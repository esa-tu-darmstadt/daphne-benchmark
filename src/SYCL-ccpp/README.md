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

  To compile all kernels in their default configuration:
  $ make

  The same result can be achieved in the kernel subfolder (points2image/eucldiean_cluster/ndt_mapping):
  $ make

  This operation requires a working Codeplay ComputeCpp installation and compute++ in PATH.
  If the system does not find the sycl headers it can be pointed to it through the Makefile variable SYCL_INCLUDE_PATH.
  The program is linked with libComputeCpp.so.
  Like with the headers the system can be pointed to it with the SYCL_LIBRARY_PATH variable:
  $ make SYCL_LIBRARY_PATH=/my/directory/lib SYCL_INCLUDE_PATH=/my/directory/include


  The compute++ tool generates device specific kernel code.
  By default the Makefile tasks it target spir64 compatible devices.
  Since not all devices run spir64 code the kernel binary format can be specified with the SYCL_KERNEL_FORMAT variable.
  It can take any value of spir, spirv, spir64, spirv64, ptx64.
  The target device is selected with the SYCL_DEVICE_TYPE variable which can be set to
  CPU, GPU or HOST.
  In combination, to explicitly target our NVIDIA Graphics card we can type:
  $ make SYCL_DEVICE_TYPE=GPU SYCL_KERNEL_FORMAT=ptx64

* Execute the benchmark

  In the kernel subfolder:
  $ ./kernel

  This will print information about the kernel runtime, unexpected deviations and errors during execution.
  The selection of an unsupported kernel format in combination with a device
  usually results in a cl::sycl::compile_program_error exception at runtime.
