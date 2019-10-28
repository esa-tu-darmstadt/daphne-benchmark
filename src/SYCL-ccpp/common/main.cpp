/**
 * Author:  Florian Stock, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2018
 * License: Apache 2.0 (see attachached File)
 */
#include <chrono>
#include <iostream>
#include <string.h>
#include <stdlib.h>
#include "benchmark.h"

// fields for runtime measurement
std::chrono::high_resolution_clock::time_point start,end;
std::chrono::duration<double> elapsed;
std::chrono::high_resolution_clock timer;
bool pause = false;
// number of testcases to process before comparison and reading the next set of test data
int pipelined = 1;
// the kernel to execute
extern kernel& myKernel;
/**
 * Pauses the timer.
 */
void pause_timer()
{
  end = timer.now();
  elapsed += (end-start);
  pause = true;
}  
/**
 * Resumes the timer.
 */
void unpause_timer() 
{
  pause = false;
  start = timer.now();
}
/**
 * Displays usage information
 */
void usage(char *exec)
{
  std::cout << "Usage: \n" << exec << " [-p N]\nOptions:\n  -p N   executes N invocations in sequence,";
  std::cout << "before taking time and check the result.\n";
  std::cout << "         Default: N=1\n";
}
int main(int argc, char **argv) {
	// parse the pipelined argument
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
	// prepare the kernel
	myKernel.set_timer_functions(pause_timer, unpause_timer);
	myKernel.init();
	// start measuring the runtime of the kernel
	start = timer.now();
	// execute the kernel
	myKernel.run(pipelined);
	// measure the runtime of the kernel
	if (!pause) 
	{
		end = timer.now();
		elapsed += end-start;
	}
	// display results
	std::cout <<  "elapsed time: "<< elapsed.count() << " seconds, average time per testcase (#"
			<< myKernel.testcases << "): " << elapsed.count() / (double) myKernel.testcases
			<< " seconds" << std::endl;
	if (myKernel.check_output())
	{
		std::cout << "result ok\n";
		return 0;
	} else 
	{
		std::cout << "error: wrong result\n";
		return 1;
	}
}
