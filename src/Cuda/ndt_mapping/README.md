* Preparation

  Extract the test case data files
  * ndt_input.dat.gz
  * ndt_output.dat.gz
  inside the data folder

  Make sure that nvcc is available
  If not already in the search path it is located at /usr/local/cuda/bin for a default CUDA installation
  In this situation the PATH variable can temporarily be altered with:

  $ export PATH=$PATH:/usr/local/cuda/bin

* Compilation

  Navigate to src/Cuda/ndt_mapping and type:

  $ make

  The CUDA target architecture can be set with the CUDA_DEVICE_ARCH argument
  To select 6.0 compute capability type:

  $ make CUDA_DEVICE_ARCH=compute_60

  The default compute capability selected is 7.2

* Running the benchmark

  Inside the src/Cuda/ndt_mapping folder:

  $ ./kernel

  This tells us about:
  * Errors opening or reading from the test case data files
  * kernel runtime
  * deviations from the expected results
