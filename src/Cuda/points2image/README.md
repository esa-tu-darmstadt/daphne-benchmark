* Preparation

  Extract the test case data files
  * p2i_input.dat.gz
  * p2i_output.dat.gz 
  inside the data folder

* Compilation

  Navigate to src/Cuda/points2image and type:

  $ make

  The CUDA target architecture can be set with the CUDA_DEVICE_ARCH argument
  To select 6.0 compute capability type:

  $ make CUDA_DEVICE_ARCH=compute_62

  The default compute capability selected is 6.2

* Running the benchmark

  Inside the src/Cuda/points2image folder:

  $ ./kernel

  This tells us about:
  * Errors opening the test case data files
  * kernel runtime
  * deviations from the expected results
