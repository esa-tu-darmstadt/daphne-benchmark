# The Darmstadt Automotive Parallel HeterogeNEous (DAPHNE) Benchmark-Suite #

This suite contains automotive benchmarks used for the evaluation of heterogeneous, parallel programming models. They are extracted from the Autoware project and should represent parallelizable workloads from the automotive field.

It contains 3 kernels (euclidean\_cluster, ndt\_matching and points2image) and four different implementations for each kernel:
-Serial version
-OpenMP version
-CUDA version
-OpenCL version (here it can be necessary to change the device constant, as some accelerators are recognized as accelerators and some as GPUs. When querying for the device, set it either to "CL\_DEVICE\_TYPE\_ACCELERATOR" or "CL\_DEVICE\_TYPE\_GPU").


Build the benchmarks either with
a) the toplevel Makefile. It defaults to build all architectures and allows to choose from opencl, cpu, openmp, or cuda.

b) a Makefile within one of the architecture folders in the src directory.

c) a Makefile in a specific kernel directory (ndt\_matching, points2image, or euclidean\_clustering).

Depending on the platform and environment it can be necessary to configure path variables in the Makefile (esp. for the OpenCL version). 

To run a benchmark, execute the ./kernel executable in each testcase directory.

Input- and golden reference data resides in the subdirectory 'data'. In this package, we provide the "minimal" data-set. We provide additional data-sets "small", "medium" and "full" with different number of testcases/sizes via download from TODO. Note that you need to unzip the files in the "data" directory before you run the kernels.

### FPGA Implementation ###

For the points2image benchmark, this release also contains an OpenCL implementation targeting Xilinx FPGAs (Zynq Ultrascale+ ZCU102) with Xilinx SDSoC/SDAccel. Due to the peculiarities of the setup, the long compilation times and the additional licenses required, building for FPGA is not include in the top-level Makefile.
