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

  The same result can be achieved in the respective kernel subfolder:
  $ cd points2image
  or $ cd euclidean_cluster
  or $ cd ndt_mapping
  $ make

  Compilation requires a working Codeplay ComputeCpp installation and compute++ in PATH.
  If the system does not find the sycl headers automatically,
  manual configuration can be achieved through the SYCL_INCLUDE_PATH variable.
  The program is linked with the libComputeCpp.so shared library.
  Like with the sycl headers, the location can be set manually with the SYCL_LIBRARY_PATH variable:
  $ make SYCL_LIBRARY_PATH=/my/directory/lib SYCL_INCLUDE_PATH=/my/directory/include

  The compute++ tool generates device specific kernel code.
  By default the Makefile tasks compute++ to target spir64 compatible devices.
  Since not all devices run spir64 code the kernel binary format can be specified with the SYCL_KERNEL_FORMAT variable.
  Valid configuration values are: spir, spirv, spir64, spirv64, ptx64
  The target device is selected with the SYCL_DEVICE_TYPE variable which can be set to
  CPU, GPU or HOST.
  Put together, explicit targeting an NVIDIA graphics card can be achieved with:
  $ make SYCL_DEVICE_TYPE=GPU SYCL_KERNEL_FORMAT=ptx64

  The selection can be further specialized by OpenCL-typical platform and device names with the
  SYCL_DEVICE_NAME and SYCL_PLATFORM_NAME variables. The resulting command may look similar to:
  $ make SYCL_PLATFORM_NAME=CUDA SYCL_DEVICE_NAME=RTX SYCL_KERNEL_FORMAT=ptx64

  Other common configuration options:
  * TESTCASE_LIMIT - limit the number of processed testcases
    - provide an integer to only process a subset of available testcases
  * ENABLE_LEGACY_TESTDATA - enable support for old test data format
    - only use this option if legacy test data is actually supplied
  * ENABLE_TESTDATA_GEN - enable reference data generation
    - generated reference  data directory which can be used as reference data in subsequent executions
  * ENABLE_SYCL_MEMCPY - enable SYCL memcpy/memset intrinsics instead of serial fallback functions
    - enabling this option passes -no-serial-memop to compute++
  * SYCL_WORK_GROUP_SIZE - specify a custom work group size for SYCL kernel calls
    - can be used as a workaround to incompatibility with default values



* Execute the benchmark

  In the kernel subfolder:
  $ ./benchmark

  This will print information about the runtime, unexpected deviations and errors during execution.

  It is common to see error outputs in case of unavailable SYCL devices, usage of unsupported kernel formats
  or missing hardware features.
