* Preparation
  Extract the test case data files
  * p2i_input.dat.gz
  * p2i_output.dat.gz
  inside the data folder
  
* Compilation
  
  Navigate to src/OpenCl/points2image:
  
  $ make
  
  Optional arguments are:
  * OPENCL_PLATFORM_ID - to select a specific platform
    - can be a platform index e.g. 1 or a identifiying string e.g. AMD
  * OPENCL_DEVICE_ID - to select a specific device by index or name
    - can be an index or an identifiying string e.g. RX 480
  * OPENCL_DEVICE_TYPE - to limit device selection to a specific device type
    - can be CPU, GPU, ACC or DEFAULT
    - CPUs only, GPUs only, other accelerator types only or default device selection
  
  For example if we wanted to select our Nvidia RTX 2070 graphics card we could type:
  
  $ make OPENCL_DEVICE_ID=RTX
  
  to select the first graphics card which name contains "RTX"
  By default the first available OpenCL capable device is selected
  
* Running the benchmark

  In the src/OpenCl/points2image folder:
  
  $ ./kernel
  
  This will inform us about:
  * whether the test data could be found
  * the OpenCL device selected
  * problems during the OpenCL setup phase
  * kernel runtime
  * deviation from the reference data
