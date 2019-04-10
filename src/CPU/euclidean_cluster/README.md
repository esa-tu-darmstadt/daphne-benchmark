* Preparation

  Extract the test case data files
  * ec_input.dat.gz
  * ec_output.dat.gz
  inside the data folder

  This step should yield
  * ec_input.dat
  * ec_output.dat
  in the data folder

* Compilation

  Navigate to the src/CPU/euclidean_cluster folder

  In a shell type:
  $ make

* Execute the benchmark

  In src/CPU/euclidean_cluster:

  $ ./kernel

  This will inform about the kernel runtime and unexpected deviations from the reference results

