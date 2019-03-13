#include <chrono>
#include <iostream>
#include <string.h>
#include <stdlib.h>
#include "benchmark.h"

std::chrono::high_resolution_clock::time_point start,end;
std::chrono::duration<double> elapsed;
std::chrono::high_resolution_clock timer;
bool pause = false;

// how many testcases should be executed in sequence (before checking for correctness)
int pipelined = 1;

extern kernel& myKernel;


void pause_timer()
{
  end = timer.now();
  elapsed += (end-start);
  pause = true;
}  

void unpause_timer() 
{
  pause = false;
  start = timer.now();
}

void usage(char *exec)
{
  std::cout << "Usage: \n" << exec << " [-p N]\nOptions:\n  -p N   executes N invocations in sequence,";
  std::cout << "before taking time and check the result.\n";
  std::cout << "         Default: N=1\n";
}


// ---------------------------------------------------------------
#if defined (OPENCL)
	#include "ocl_header.h"

OCL_Struct OCL_objs;
#endif
// ---------------------------------------------------------------

int main(int argc, char **argv) {

  if ((argc != 1) && (argc !=  3))
    {
      usage(argv[0]);
      exit(2);
    }
  if (argc == 3)
    {
      if (strcmp(argv[1], "-p") != 0)
	{
	  usage(argv[0]);
	  exit(3);
	}
      errno = 0;
      pipelined = strtol(argv[2], NULL, 10);
      if (errno || (pipelined < 1) )
	{
	  usage(argv[0]);
	  exit(4);
	}
      std::cout << "Invoking kernel " << pipelined << " time(s) per measure/checking step\n";
      
    }
    // read input data
    myKernel.set_timer_functions(pause_timer, unpause_timer);
    myKernel.init();

	// ---------------------------------------------------------------
#if defined (OPENCL)

	// Define OpenCL platform and needed objects
	// DO NOT USE CPP BINDINGS
	cl_int err;
	cl_platform_id* local_platform_id;
	cl_uint         local_platformCount;

	err = clGetPlatformIDs(0, NULL, &local_platformCount);

	if (err != CL_SUCCESS){
		printf("Error: clGetPlatformIDs(): %d\n",err);
		fflush(stdout);
 		return EXIT_FAILURE;
  	}
	local_platform_id = (cl_platform_id*) malloc(sizeof(cl_platform_id) * local_platformCount);

	err = clGetPlatformIDs(local_platformCount, local_platform_id, NULL);
  	if (err != CL_SUCCESS){
		printf("Error: clGetPlatformIDs(): %d\n",err);
		fflush(stdout);
	 	return EXIT_FAILURE;
  	}

	OCL_objs.rcar_platform = local_platform_id;

	// retrieving the CVEngine device
	err = clGetDeviceIDs(OCL_objs.rcar_platform[0], CL_DEVICE_TYPE_ACCELERATOR, 1, & OCL_objs.cvengine_device, NULL);

	// creating the context for RCar OpenCL devices
	OCL_objs.rcar_context =  clCreateContext(0, 1, &OCL_objs.cvengine_device, NULL, NULL, &err);

	// creating a command queueu for CVEngine accelerator
	OCL_objs.cvengine_command_queue = clCreateCommandQueue(OCL_objs.rcar_context, OCL_objs.cvengine_device, CL_QUEUE_PROFILING_ENABLE, &err);


#if 0
    try {
	// Get list of OpenCL platforms.
	std::vector<cl::Platform> platform;
	cl::Platform::get(&platform);

	if (platform.empty()) {
	    std::cerr << "OpenCL platforms not found." << std::endl;
	    return 1;
	}

	// Get first available GPU device which supports double precision.
	cl::Context context;
	std::vector<cl::Device> device;
	for(auto p = platform.begin(); device.empty() && p != platform.end(); p++) {
	    std::vector<cl::Device> pldev;

	    try {
		/*p->getDevices(CL_DEVICE_TYPE_GPU, &pldev);*/
		p->getDevices(CL_DEVICE_TYPE_ACCELERATOR, &pldev);

		for(auto d = pldev.begin(); device.empty() && d != pldev.end(); d++) {
		    if (!d->getInfo<CL_DEVICE_AVAILABLE>()) continue;

		    std::string ext = d->getInfo<CL_DEVICE_EXTENSIONS>();

/*
		    if (
			    ext.find("cl_khr_fp64") == std::string::npos &&
			    ext.find("cl_amd_fp64") == std::string::npos
		       ) continue;
*/

		    device.push_back(*d);
		    context = cl::Context(device);
		}
	    } catch(...) {
		device.clear();
	    }
	}

	if (device.empty()) {
	    std::cerr << "GPUs with double precision not found." << std::endl;
	    return 1;
	}

	std::cout << "GPU device: " << device[0].getInfo<CL_DEVICE_NAME>() << std::endl;

	// Create command queue.
	cl::CommandQueue queue(context, device[0]);

	// Preparing data to pass to kernel functions
	OCL_objs.platform = platform[0];
	OCL_objs.device   = device[0];
	OCL_objs.context  = context;
  	OCL_objs.cmdqueue = queue;

	// Kernel launch is specific, and therefore
	// takes places in each corresponding kernel.cpp file

    } catch (const cl::Error &err) {
	std::cerr
	    << "OpenCL error: "
	    << err.what() << "(" << err.err() << ")"
	    << std::endl;
	return 1;
    }
#endif // end of #if 0
#endif
	// ---------------------------------------------------------------

    
    // measure the runtime of the kernel
    start = timer.now();

    // execute the kernel
    myKernel.run(pipelined);

#if defined (OPENCL)
	err = clReleaseCommandQueue(OCL_objs.cvengine_command_queue);
	err = clReleaseContext(OCL_objs.rcar_context);


#endif


  
    // measure the runtime of the kernel
    if (!pause) 
    {
	end = timer.now();
    	elapsed += end-start;
    }
    std::cout <<  "elapsed time: "<< elapsed.count() << " seconds, average time per testcase (#"
	      << myKernel.testcases << "): " << elapsed.count() / (double) myKernel.testcases
	      << " seconds" << std::endl;

    // read the desired output  and compare
    if (myKernel.check_output())
      {
	std::cout << "result ok\n";
	return 0;
      } else
      {
	std::cout << "error: wrong result\n";
	return 1;
      }
    return 1;

}
