* Preparation

  Extract the test case data files
  * p2i_input.dat.gz
  * p2i_output.dat.gz 
  inside the data folder
  
  This step should yield 
  * p2i_input.dat
  * p2i_output.dat
  in the data folder

* Compilation

  Navigate to the src/CPU/points2image folder
  
  In a shell type:
  $ make

* Execute the benchmark

  In src/CPU/points2image:
  
  $ ./kernel
  
  This will inform about the kernel runtime and unexpected deviations from the reference results
  
