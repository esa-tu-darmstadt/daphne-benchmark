/**
 * Author:  Florian Stock, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2018
 * License: Apache 2.0 (see attached files)
 */
#include <chrono>
#include <iostream>
#include <string.h>
#include <stdlib.h>

#include "benchmark.h"

extern benchmark& myKernel;

int main(int argc, char **argv) {
	myKernel.init();
	// execute the kernel
	int pipelined = 1;
	myKernel.run(pipelined);
	// measure the runtime of the kernel
	myKernel.quit();
	double secondsElapsed = myKernel.get_runtime();
	int testcaseNo = myKernel.get_testcase_no();
	if (testcaseNo > 0) {
		std::cout <<  "elapsed time: "<< secondsElapsed << " seconds";
		std::cout <<  ", average time per testcase (#"<< testcaseNo << "): ";
		std::cout << secondsElapsed/(double)testcaseNo << " seconds" << std::endl;
	}

	// read the desired output  and compare
	if (myKernel.check_output()) {
		std::cout << "result ok\n";
		return 0;
	} else {
		std::cout << "error: wrong result\n";
		return 1;
	}
}
