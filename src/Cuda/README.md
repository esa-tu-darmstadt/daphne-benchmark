* Preparation

  Extract the test data files inside the data folder (../../data)
  Test data is supplied separately for each benchmark.

  For example the points2image benchmark accesses the following data files:
  * p2i_input.dat
  * p2i_output.dat

  These can be extracted inside the data folder with:
  $ gunzip --keep p2i_input.dat.gz p2i_input.dat.gz

  Make sure that nvcc is available
  If not already in the search path it is located at /usr/local/cuda/bin for a default CUDA installation
  In this situation the PATH variable can be temporarily altered with:
  $ export PATH=$PATH:/usr/local/cuda/bin

* Compilation

  To compile all kernels in their default configuration type:
  $ make

  For separate compilation with the possibility to specify additional arguments
  head to the respective benchmark subfolder and call make:
  $ cd points2image
  $ make

  Common configuration options are:
  * CUDA_BLOCK_SIZE - set the block size in kernel calls
    - must be a value supported by the targeted device
    - the default is 512 or 256 depending on the benchmark
  * CUDA_DEVICE_ARCH - set the target architecture for kernel code
    - provide a value in the format compute_XY
    - the architecture must be supported by the target device
  * TESTCASE_LIMIT - limit the number of testcases processed
    - used to process only the first n test cases in the test data

  Additional options for the points2image benchmark:
  * ENABLE_TESTDATA_LEGACY - alternative, packed test data representation
    - compatibility switch for usage with an old test data format
    - increases load times, only enable if legacy test data is actually provided

  Example with explicit block size and testcase limit customization:
  $ make CUDA_BLOCK_SIZE=256 TESTCASE_LIMIT=231


* Running the benchmark

  Execute the benchmark from within its subfolder:
  $ cd points2image/
  $ ./benchmark

  This outputs information about:
  * Errors opening or reading from the test data files
  * benchmark runtime
  * deviations from the expected results
