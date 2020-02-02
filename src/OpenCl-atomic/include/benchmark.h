/**
 * Author:  Florian Stock, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2018
 * License: Apache 2.0 (see attachached File)
 */
#ifndef benchmark_h
#define benchmark_h

#include <iostream>

class kernel {
public:
	// the number of available test cases
	uint32_t testcases;
protected:
	void (*unpause_func)();
	void (*pause_func)();
protected:
	kernel() :
		testcases(1),
		unpause_func(nullptr),
		pause_func(nullptr)
	{}
	virtual ~kernel() {};
public:
	/*
	* Initializes the kernel. Must be called before run().
	*/
	virtual void init() = 0;
	/**
	 * Performs the kernel operations on all input and output data.
	 * p: number of testcases to process in one step
	 */
	virtual void run(int p = 1) = 0;
	/**
	 * Deinitializes the kernel. To be called before program exit.
	 */
	virtual void quit() = 0;
	/**
	 * Finally checks whether all input data has been processed successfully.
	 */
	virtual bool check_output() = 0;
	/* Sets the function to call for pausing and resuming time measurements.
	 * pause_function: timer pause function
	 * unpause_function: timer unpause function
	 */
	void set_timer_functions(void (*pause_function)(), void (*unpause_function)()) {
		unpause_func = unpause_function;
		pause_func = pause_function;
	}
protected:
	virtual int read_next_testcases(int count) = 0;
};

#endif
