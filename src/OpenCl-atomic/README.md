* Preparation

  Extract the test data files inside the data folder (../../data)
  Test data is supplied separately for each benchmark.

  For example the points2image benchmark accesses the following data files:
  * p2i_input.dat
  * p2i_output.dat

  These can be extracted inside the data folder with:
  $ gunzip --keep p2i_input.dat.gz p2i_input.dat.gz

* Compilation

  Compilation for all OpenCL benchmarks with default configurations can be initiated from this directory by calling make:
  $ make

  Compilation with non standard configurations is possible by calling make in the respective subfolder:
  $ cd points2image
  $ make

  Common configuration options are:
  * OPENCL_PLATFORM_ID - to select a specific platform
    - can be a platform index e.g. 1 or a identifiying string e.g. AMD
    - by default the first availble OpenCL platform is selected
  * OPENCL_DEVICE_ID - to select a specific device by index or name
    - can be an index or an identifiying string e.g. RX 480
    - by default the first available OpenCL device is selected
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
    - provide an integer to only process a subset of testcases
    - checked against the actual limit imposed by the test data at runtime
  * ENABLE_OPENCL_ZERO_COPY - enable host side buffers and memory mapping
    - can decrease runtime on systems where GPU and CPU share address spaces
  * ENABLE_OPENCL_PINNED_MEMORY - enable pinned memory
    - can decrease runtime by improving memory bandwidth
  * DISABLE_OPENCL_ATOMICS - disable atomic function usage in kernels
    - compatibility switch for accelerators that do not support atomic operations

  Additional options for the points2image benchmark:
  * DISABLE_OPENCL_LOCAL_ATOMICS - only use atomics on global data
    - set to 1 to disable atomics on local data
  * OPENCL_TRANSFORMS_PER_WORK_ITEM - increases work load per work item while the number of work items decreases
    - set to 2 or more to enable this optimization
  * ENABLE_TESTDATA_LEGACY - alternative, packed test data representation
    - compatibility switch for usage with an old test data format
    - increases load times, only enable if legacy test data is actually provided


  Additional options for the euclidean_clustering benchmark:
  * OPENCL_DISTANCES_PER_PACKET - store more than one binary value in one memory location
    - can be set to 8, 16, 32 or 64 to reduce memory usage and overhead

  Additional options for the ndt_mapping benchmark:
  * ENABLE_OPENCL_CLOUD_MEASURE - use OpenCL to determine point cloud extends
    - set to 1 to enable accelerated point cloud measurement
  * OPENCL_VOXEL_POINT_STORAGE - trades memory usage for potentially better caching
    - provide the number of points that a voxel is able to store
    - keep this value low, e.g. below 5, to keep memory usage within sensible bounds

  Information about enabled an disabled features is output during compilation.
  Example to select an NVIDIA RTX series graphcis card and set the test case limit to 100:
  $ make OPENCL_DEVICE_ID=RTX TESTCASE_LIMIT=100
  With that the benchmark selects the first graphics card which name contains "RTX".
  Recompilation may be required to pass new arguments to already compiled targets. It can be forced with:
  $ make --always-make

* Running the benchmark

  Execute the benchmark from within the benchmark subfolder:
  $ cd points2image
  $ ./benchmark

  This will output information about:
  * whether the test data could be found
  * the OpenCL device selected
  * problems during the OpenCL setup phase
  * benchmark runtime
  * deviation from the reference data
