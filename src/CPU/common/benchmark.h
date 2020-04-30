/**
 * Author:  Florian Stock, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2018
 * License: Apache 2.0 (see attached files)
 */
#ifndef EPHOS_BENCHMARK_H
#define EPHOS_BENCHMARK_H

#include <iostream>
#include <chrono>

class benchmark {
private:
	std::chrono::high_resolution_clock::time_point timeStart,timeEnd;
	std::chrono::duration<double> timeElapsed;
	std::chrono::high_resolution_clock timer;
	bool timerPaused;
protected:
	uint32_t testcases;
protected:
	benchmark() :
		testcases(1),
		timeStart(),
		timeEnd(),
		timeElapsed(0.0),
		timer(),
		timerPaused(true)
	{}
	virtual ~benchmark() {};
public:
	/*
	* Initializes the benchmark. Must be called before run().
	*/
	virtual void init() = 0;
	/**
	 * Performs the benchmark operations on all input and output data.
	 * p: number of testcases to process in one step
	 */
	virtual void run(int p = 1) = 0;
	/**
	 * Deinitializes the benchmark. To be called before program exit.
	 */
	virtual void quit() = 0;
	/**
	 * Finally checks whether all input data has been processed successfully.
	 */
	virtual bool check_output() = 0;
	/**
	 * Gets the number of seconds the benchmark took to execute.
	 */
	double get_runtime() {
		return timeElapsed.count();
	}
	/**
	 * Gets the number of test cases in the test data.
	 */
	int get_testcase_no() {
		return testcases;
	}
protected:
	/**
	 * Starts the benchmark timer.
	 * Should be called before processing the test data.
	 */
	void start_timer() {
		timeElapsed = std::chrono::duration<double>(0.0);
		timerPaused = true;
		resume_timer();
	}
	/**
	 * Stops the benchmark timer.
	 * Should be called when all test cases have been processed.
	 */
	double stop_timer() {
		pause_timer();
	}
	/**
	 * Pauses the benchmark timer.
	 * Should be called in between test cases.
	 */
	void pause_timer() {
		if (!timerPaused) {
			timeEnd = timer.now();
			timeElapsed += timeEnd - timeStart;
			timerPaused = true;
		}
	}
	/**
	 * Resumes the benchmark timer after a pause.
	 * Should be called at the beginning of a test case.
	 */
	void resume_timer() {
		if (timerPaused) {
			timeStart = timer.now();
			timerPaused = false;
		}
	}
};

#endif
