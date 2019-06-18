* Preparation

  Extract the test case data files inside the data folder
  
  For points2image these are
  * p2i_input.dat.gz
  * p2i_output.dat.gz
  
  For euclidean_cluster these are
  * ec_input.dat.gz
  * ec_output.dat.gz
  
  For ndt_mapping these are
  * ndt_input.dat.gz
  * ndt_output.dat.gz
  
  Example:
  $ gunzip --keep ec_input.dat.gz ec_output.dat.gz
  
  Which yields:
  * ec_input.dat
  * ec_output.dat 
  inside the data folder

* Compilation

  Make sure to use a compiler that supports offloading with OpenMP
  Non offloading kernel variants are available in the src/OpenMP folder

  Compilation for all OpenMP-offload kernels can be started from this folder with:
  $ make

  They kernels can also be individually compiled with:
  $ make 
 
  in their respective subfolders (points2image, euclidean_cluster, ndt_mapping)
  
  To use a specific compiler set the CXX variable:
  $ make CXX=/path/to/compiler/executable
  
  The automatically selected host and target devices might not match your hardware.
  The device IDs used for buffer allocation and memory copies can be set with OPENMP_TARGET_DEVICE_ID and OPENMP_HOST_DEVICE_ID:
  $ make OPENMP_HOST_DEVICE_ID=-2 OPENMP_TARGET_DEVICE_ID=0

* Execute the benchmark

  In the respective kernel subfolder type:
  $ ./kernel

  This will given information about offloading devices, kernel runtime, unexpected deviations from the reference results
