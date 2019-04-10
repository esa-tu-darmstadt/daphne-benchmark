* Preparation

  Extract the test case data files
  * ec_input.dat.gz
  * ec_output.dat.gz
  inside the data folder

  Make sure that nvcc is available
  If not already in the search path it is located at /usr/local/cuda/bin for a default CUDA installation
  In this situation the PATH variable can temporarily be altered with:

  $ export PATH=$PATH:/usr/local/cuda/bin

* Compilation

  Navigate to src/Cuda/eucldean_cluster and type:

  $ make

  The CUDA target architecture can be set with the CUDA_DEVICE_ARCH argument
  To select 6.0 compute capability type:

  $ make CUDA_DEVICE_ARCH=compute_60

  The default compute capability selected is 6.0

* Running the benchmark

  Inside the src/Cuda/euclidean_cluster folder:

  $ ./kernel

  This tells us about:
  * Errors opening or reading from the test case data files
  * kernel runtime
  * deviations from the expected results
