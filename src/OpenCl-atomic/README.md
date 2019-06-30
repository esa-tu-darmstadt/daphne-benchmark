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

  From there optional arguments can be supplied:
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
  * OPENCL_LOCAL_SIZE - to select a specific work group size
    - number of work items in a work group, e.g. 512

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
