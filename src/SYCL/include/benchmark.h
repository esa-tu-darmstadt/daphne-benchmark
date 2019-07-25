#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <iostream>

class kernel {
public:
	// the number of testcase available for this kernel (there should be at least 1)
	uint32_t testcases = 1;
	
	/**
	 * Performs necessary pre-run initialisation. It usually
	 * reads the necessary input data from a file.
	 */
	virtual void init() = 0;

	/**
	 * Executes the testcase, in blocks of p at a time.
	 * p: the number of testcases to process
	 */
	virtual void run(int p = 1) = 0;

	/**
	 * Compares the computed output with the reference data.
	 */
	virtual bool check_output() = 0;

	/* 
	 * Sets the functions to call for pausing and resuming runtime measurement.
	 */
	void set_timer_functions(
		void (*pause_function)(),
		void (*unpause_function)()) {
		unpause_func = unpause_function;
		pause_func = pause_function;
	}

protected:
	// the function to call for resuming runtime measurement
	void (*unpause_func)();
	// the function to call for pausing runtime measurement
	void (*pause_func)();
	
	/**
	 * Reads the next testcases.
	 * count: the number of testcases to read
	 * return: the number of testcases actually read
	 */
	virtual int read_next_testcases(int count) = 0;
};

#endif
