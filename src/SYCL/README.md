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

* Compilation

  To compile all kernels in their default configuration:
  $ make

  The same result can be achieved in the kernel subfolder (points2image/eucldiean_cluster/ndt_mapping):
  $ make

* Execute the benchmark

  In the kernel subfolder:
  $ ./kernel

  This will print information about the kernel runtime and unexpected deviations from the reference results
