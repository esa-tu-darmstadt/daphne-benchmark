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

  Make sure that nvcc is available
  If not already in the search path it is located at /usr/local/cuda/bin for a default CUDA installation
  In this situation the PATH variable can temporarily be altered with:
  $ export PATH=$PATH:/usr/local/cuda/bin

* Compilation

  To compile all kernels in their default configuration type:
  $ make

  For separate compilation navigate to the kernel subfolder (points2image/euclidean_cluster/ndt_mapping) and type:
  $ make

  The CUDA target architecture can be set with the CUDA_DEVICE_ARCH argument
  To select 6.2 compute capability type:
  $ make CUDA_DEVICE_ARCH=compute_62

  The default compute capability selected is 6.2

* Running the benchmark

  Inside the kernel subfolder:
  $ ./kernel

  This outputs information about:
  * Errors opening or reading from the test case data files
  * kernel runtime
  * deviations from the expected results
