#include "benchmark.h"
#include "datatypes.h"
#include "kernel.h"
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <limits>
#include <cstring>
#include <chrono>

Matrix4f Matrix4f_Identity = {
	{{1.0, 0.0, 0.0, 0.0},
	 {0.0, 1.0, 0.0, 0.0},
	 {0.0, 0.0, 1.0, 0.0},
	 {0.0, 0.0, 0.0, 1.0}}
};

/**
 * Reads the next point cloud.
 */
void  parseFilteredScan(std::ifstream& input_file, PointCloud* pointcloud) {
	int32_t size;
	try {
		input_file.read((char*)&(size), sizeof(int32_t));
		pointcloud->clear();
		for (int i = 0; i < size; i++)
		{
			PointXYZI p;
			input_file.read((char*)&p.data[0], sizeof(float));
			input_file.read((char*)&p.data[1], sizeof(float));
			input_file.read((char*)&p.data[2], sizeof(float));
			input_file.read((char*)&p.data[3], sizeof(float));
			pointcloud->push_back(p);
		}
	}  catch (std::ifstream::failure) {
		throw std::ios_base::failure("Error reading filtered scan");
	}
}

/**
 * Reads the next initilization matrix.
 */
void  parseInitGuess(std::ifstream& input_file, Matrix4f* initGuess) {
	try {
	for (int h = 0; h < 4; h++)
		for (int w = 0; w < 4; w++)
			input_file.read((char*)&(initGuess->data[h][w]),sizeof(float));
	}  catch (std::ifstream::failure) {
		throw std::ios_base::failure("Error reading initial guess");
	}
}

/**
 * Reads the next reference matrix.
 */
void parseResult(std::ifstream& output_file, CallbackResult* goldenResult) {
	try {
		for (int h = 0; h < 4; h++)
			for (int w = 0; w < 4; w++)
			{
				output_file.read((char*)&(goldenResult->final_transformation.data[h][w]), sizeof(float));
			}
		output_file.read((char*)&(goldenResult->fitness_score), sizeof(double));
		output_file.read((char*)&(goldenResult->converged), sizeof(bool));
	}  catch (std::ifstream::failure) {
		throw std::ios_base::failure("Error reading reference result");
	}
}

int ndt_mapping::read_next_testcases(int count)
{
	int i;
	// free memory used in the previous test case and allocate new one
	delete [] maps;
	maps = new PointCloud[count];
	delete [] filtered_scan_ptr;
	filtered_scan_ptr = new PointCloud[count];
	delete [] init_guess;
	init_guess = new Matrix4f[count];
	delete [] results;
	results = new CallbackResult[count];
	// parse the test cases
	for (i = 0; (i < count) && (read_testcases < testcases); i++,read_testcases++)
	{
		try {
			parseInitGuess(input_file, init_guess + i);
			parseFilteredScan(input_file, filtered_scan_ptr + i);
			parseFilteredScan(input_file, maps + i);
		} catch (std::ios_base::failure& e) {
			std::cerr << e.what() << std::endl;
			exit(-3);
		}
	}
	return i;
}


int ndt_mapping::read_number_testcases(std::ifstream& input_file)
{
	int32_t number;
	try {
		input_file.read((char*)&(number), sizeof(int32_t));
	} catch (std::ifstream::failure) {
		throw std::ios_base::failure("Error reading number of test cases");
	}
	return number;
}

void ndt_mapping::check_next_outputs(int count)
{
	CallbackResult reference;
	for (int i = 0; i < count; i++)
	{
		try {
			parseResult(output_file, &reference);
		} catch (std::ios_base::failure& e) {
			std::cerr << e.what() << std::endl;
			exit(-3);
		}
		std::ostringstream sError;
		int caseErrorNo = 0;
		if (results[i].converged != reference.converged)
		{
			error_so_far = true;
			caseErrorNo += 1;
			if (reference.converged) {
				sError << " computation converged invalidly" << std::endl;
			} else {
				sError << " computation did not converge" << std::endl;
			}
		}
		// compare the matrices
		for (int h = 0; h < 4; h++) {
			// test for nan
			for (int w = 0; w < 4; w++) {
				if (std::isnan(results[i].final_transformation.data[h][w]) !=
					std::isnan(reference.final_transformation.data[h][w])) {

					error_so_far = true;
					caseErrorNo += 1;
					sError << " matrix at [" << h << " " << w << "]: ";
					sError << reference.final_transformation.data[h][w] << " should be ";
					sError << results[i].final_transformation.data[h][w] << std::endl;
				}
			}
			// compare translation
			float delta = std::fabs(results[i].final_transformation.data[h][3] -
				reference.final_transformation.data[h][3]);
			if (delta > MAX_TRANSLATION_EPS) {
				error_so_far = true;
				caseErrorNo += 1;
				sError << " matrix at [" << h << " " << 3 << "]: ";
				sError << reference.final_transformation.data[h][3] << " should be ";
				sError << results[i].final_transformation.data[h][3] << std::endl;
			}
			if (delta > max_delta) {
				max_delta = delta;
			}
		}
		// compare transformed points
		PointXYZI origin = {
			{ 0.724f, 0.447f, 0.525f, 1.0f }
		};
		PointXYZI resPoint = {
			{ 0.0f, 0.0f, 0.0f, 0.0f }
		};
		PointXYZI refPoint = {
			{ 0.0f, 0.0f, 0.0f, 0.0f }
		};
		for (int h = 0; h < 4; h++) {
			for (int w = 0; w < 4; w++) {
				resPoint.data[h] += results[i].final_transformation.data[h][w]*origin.data[w];
				refPoint.data[h] += reference.final_transformation.data[h][w]*origin.data[w];
			}
		}
		for (int w = 0; w < 4; w++) {
			float delta = std::fabs(resPoint.data[w] - refPoint.data[w]);
			if (delta > max_delta) {
				max_delta = delta;
			}
			if (delta > MAX_ROTATION_EPS) {
				error_so_far = true;
				caseErrorNo += 1;
				sError << " vector at [" << w << "]: ";
				sError << resPoint.data[w] << " should be " << refPoint.data[w] << std::endl;
			}
		}
		if (caseErrorNo > 0) {
			std::cout << "Errors for test case " << read_testcases - count + i;
			std::cout << " (" << caseErrorNo << "):" << std::endl;
			std::cout << sError.str() << std::endl;
		}

// 		std::cout << "Test case matrix Reference " << read_testcases - count + i << std::endl;
// 		for (int h = 0; h < 4; h++) {
// 			for (int w = 0; w < 4; w++) {
// 				std::cout << " " << reference.final_transformation.data[h][w];
// 			}
// 			std::cout << std::endl;
// 		}
// 		std::cout << "Test case matrix Result " << read_testcases -count + i << std::endl;
// 		for (int h = 0; h < 4; h++) {
// 			for (int w = 0; w < 4; w++) {
// 				std::cout << " " << results[i].final_transformation.data[h][w];
// 			}
// 			std::cout << std::endl;
// 		}
// 		std::cout << std::endl;
	}
}

bool ndt_mapping::check_output() {
	std::cout << "checking output \n";
	// complement to init()
	input_file.close();
	output_file.close();
	// check for error
	std::cout << "max delta: " << max_delta << "\n";
	return !error_so_far;
}
