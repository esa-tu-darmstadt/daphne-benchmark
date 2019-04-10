* Preparation

  Extract the test case data files
  * ndt_input.dat.gz
  * ndt_output.dat.gz
  inside the data folder

  This step should yield
  * ndt_input.dat
  * ndt_output.dat
  in the data folder

* Compilation

  Navigate to the src/OpenMP/ndt_mapping folder

  In a shell type:
  $ make

* Execute the benchmark

  In src/OpenMP/ndt_mapping:

  $ ./kernel

  This will inform about the kernel runtime and unexpected deviations from the reference results
