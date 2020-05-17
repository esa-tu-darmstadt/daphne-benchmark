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
  * TESTCASE_LIMIT - to limit the number of processed testcases
    - provide an integer to only process a subset of testcases
    - checked against the actual limit imposed by the test data at runtime

  Additional options for the points2image benchmark:
  * TESTCASE_SPARSE - enable alternative, packed test data representation
    - decreases load times, only applicable if sparse test data is actually provided

  Arguments can be supplied to make as follows:
  $ cd points2image
  $ make TESTCASE_LIMIT=1250 TESTCASE_SPARSE=1
  Recompilation may be required to account for updated arguments. Recompilation can be forced with:
  $ make --always-make

* Execution

  Execute the benchmark from within the benchmark subfolder:
  $ cd points2image
  $ ./benchmark

  This will output information about:
  * whether the test data could be found
  * benchmark runtime
  * deviation from the reference data