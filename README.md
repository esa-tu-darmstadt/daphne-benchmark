# The Darmstadt Automotive Parallel HeterogeNEous (DAPHNE) Benchmark-Suite #

This suite contains automotive benchmarks used for the evaluation of heterogeneous, parallel programming models. They are extracted from the Autoware project and should represent parallelizable workloads from the automotive field.

It contains 3 kernels (euclidean\_cluster, ndt\_matching and points2image) and four different implementations for each kernel:
-Serial version
-OpenMP version
-OpenMP Offloading version
-CUDA version
-OpenCL version
-OpenCL version using atomics


Build the benchmarks either with
a) the toplevel Makefile. It defaults to build all architectures and allows to choose from opencl, cpu, openmp, or cuda.

b) a Makefile within one of the architecture folders in the src directory.

c) a Makefile in a specific kernel directory (ndt\_matching, points2image, or euclidean\_clustering).

Depending on the platform and environment it can be necessary to configure path variables in the Makefile. 

To run a benchmark, execute the ./kernel executable in each testcase directory.

Input- and golden reference data resides in the subdirectory 'data'. In this package, we provide the "minimal" data-set. We provide additional data-sets "small", "medium" and "full" with different number of testcases/sizes via download from TODO. Note that you need to unzip the files in the "data" directory before you run the kernels.

### FPGA Implementation ###

For the points2image benchmark, this release also contains an OpenCL implementation targeting Xilinx FPGAs (Zynq Ultrascale+ ZCU102) with Xilinx SDSoC/SDAccel. Due to the peculiarities of the setup, the long compilation times and the additional licenses required, building for FPGA is not include in the top-level Makefile.

### Compatibility Overview ###

The benchmarks have been developed for a large number of platforms in mind. However compatibility out of the box can not be guaranteed. Since some platforms require manual configuration or do not run the benchmarks at all we provide information about which results to expect.

| Benchmark     | Consumer Desktop | Linux Workstation |
| ------------- |:----------------:| -----------------:|
| col 3 is      | right-aligned    | $1600             |
| col 2 is      | centered         |   $12             |
| zebra stripes | are neat         |    $1             |

| Compatibility                    | Platform         |                   |                    |
| Benchmark                        | Consumer Desktop | Linux Workstation | Nvidia Jetson TX 2 |
| -------------------------------- | :--------------: | :---------------: | -----------------: |
| CPU/points2image                 | ok               | ok                | ok                 |
| CPU/euclidean_cluster            | ok               | ok                | ok                 |
| CPU/ndt_mapping                  | ok               | ok                | ok                 |
|                                  |                  |                   |                    |
| Cuda/points2image                | ok               | ok (1)            | ok (2)             |
| Cuda/euclidean_cluster           | ok (1)           | ok (2)            |                    |
| Cuda/ndt_mapping                 | failed (3)       | failed (3)(1)     | ok (4)(2)          |
|                                  |                  |                   |                    |
| OpenCl/points2image              | ok               | ok                | ok (5)             |
| OpenCl/euclidean_cluster         | ok               | ok                | ok (5)             |
| OpenCl/ndt_mapping               | ok (4)           | ok (4)            | failed (5)(6)      |
|                                  |                  |                   |                    |
| OpenCl-atomic/points2image       | ok               | ok                | ok (5)             |
| OpenCl-atomic/euclidean_cluster  | ok               | ok                | ok (5)             |
| OpenCl-atomic/ndt_mapping        | ok (4)           | ok (4)            | ok (5)(4)          |
|                                  |                  |                   |                    |
| OpenMP/points2image              | ok               | ok                | ok                 |
| OpenMP/eulidean_cluster          | ok               | ok                | ok                 |
| OpenMP/ndt_mapping               | ok               | ok                | ok (4)             |
|                                  |                  |                   |                    |
| OpenMP-offload/points2image      | ok (CPU)         | ok (CPU), ok (CUDA)| ok (CPU)           |
| OpenMP-offload/euclidean_cluster | ok (CPU)         | ok (CPU), ok (CUDA)| failed (8)         |
| OpenMP-offload/ndt_mapping       | failed (CPU)(7)  | ok (CPU), ok (CUDA)(4)| failed (8)      |

(1) Compute Capability set to 6.0 or lower
(2) Compute Capability set to 6.2 or lower
(3) Results outside error tolerances
(4) Results not accurate but inside error tolerances
(5) Running on POCL with CUDA support
(6) cl_khr_int64_base_atomics not supported
(7) internal compilation error
(8) undeclared omp_target functions

