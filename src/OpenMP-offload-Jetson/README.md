## Preparation ##

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

## Compilation ##

  This version of the benchmark is specialized for OpenMP offloading on the Nvidia Jetson platforms.
  It requires to download and build a modified version of Clang & LLVM's OpenMP infrastructure that 
  you can find on [Github](https://github.com/sommerlukas/llvm-offload-jetson/tree/omp-jetson-11).
  
  First, build the LLVM project found in the repository mentioned above. Then, make sure to 
  adapt the paths in the respective Makefiles to your setup, where `LLVM_DIR` is the build 
  output directory of your compiled LLVM and `LLVM_SRC` is the LLVM project source directory.
  
  After that, you can simply build the applications with `make`. 

## Execute the benchmark ##

  In the respective kernel subfolder type:
  $ ./kernel

  This will given information about offloading devices, kernel runtime,
  unexpected deviations from the reference results
