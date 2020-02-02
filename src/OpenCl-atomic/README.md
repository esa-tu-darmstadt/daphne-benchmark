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

  Compilation for all OpenCL kernels in the default configuration can be started from this folder:
  $ make

  They can also be compiled separately in their respective subfolder with:
  $ make

  Optional arguments can be supplied from there:
  * OPENCL_PLATFORM_ID - to select a specific platform
    - can be a platform index e.g. 1 or a identifiying string e.g. AMD
  * OPENCL_DEVICE_ID - to select a specific device by index or name
    - can be an index or an identifiying string e.g. RX 480
  * OPENCL_DEVICE_TYPE - to limit device selection to a specific device type
    - can be CPU, GPU, ACC or DEFAULT
    - CPUs only, GPUs only, other accelerator types only or default device selection
  * OPENCL_INCLUDE_PATH - if the OpenCL headers are not available at the default locations
    - folder that contains CL/cl.h
  * OPENCL_LIBRARY_PATH - if the OpenCL library is not available at the default locations
    - folder that contains libOpenCL.so or similar
  * OPENCL_WORK_GROUP_SIZE - to select a specific work group size
    - number of work items in a work group, e.g. 512
  * TESTCASE_LIMIT - to limit the number of processed testcases
    - can be set between 1 and 2500 to only process a subset of testcases
  * ENABLE_OPENCL_ZERO_COPY - enable zero copy buffer operation
    - can decrease runtime on systems where GPU and CPU share address spaces
  * ENABLE_OPENCL_PINNED_MEMORY - enable pinned memory
    - can decrease runtime by improving memory bandwidth
  * DISABLE_OPENCL_ATOMICS - disable atomic function usage in kernels
    - compatibility switch for accelerators that do not support atomic operations
  * DISABLE_OPENCL_LOCAL_ATOMICS - potential optimization
    - set to 1 to disable this optimization
  * OPENCL_TRANSFORMS_PER_WORK_ITEM - increases work load per work item while the number of work items decreases
    - set to 2 or more to enable this optimization

  For example if we wanted to select our Nvidia RTX series graphics card we could type:
  $ make OPENCL_DEVICE_ID=RTX
  to select the first graphics card which name contains "RTX"
  By default the first available OpenCL capable device is selected

* Running the benchmark

  In the kernel subfolder (points2image/euclidean_cluster/ndt_mapping):
  $ ./kernel

  This will output information about:
  * whether the test data could be found
  * the OpenCL device selected
  * problems during the OpenCL setup phase
  * kernel runtime
  * deviation from the reference data
