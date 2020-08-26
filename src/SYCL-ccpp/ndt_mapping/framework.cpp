/**
 * Author:  Florian Stock, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2018
 * Author:  Thilo Gabel, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2019
 * License: Apache 2.0 (see attached files)
 */


#include <cmath>
#include <iostream>
#include <sstream>
#include <fstream>
#include <limits>
#include <cstring>
#include <chrono>
#include <stdexcept>
#include <vector>

#include "ndt_mapping.h"
#include "datatypes.h"

#include "common/benchmark.h"

int ndt_mapping::read_next_testcases(int count)
{
	int i;
	// free memory used in the previous test case and allocate new one
	maps.resize(count);
	filtered_scan.resize(count);
	init_guess.resize(count);
	results.resize(count);
	// parse the test cases
	for (i = 0; (i < count) && (read_testcases < testcases); i++,read_testcases++)
	{
		try {
			parseInitGuess(input_file, init_guess[i]);
			parseFilteredScan(input_file, filtered_scan[i]);
			parseFilteredScan(input_file, maps[i]);
		} catch (std::ios_base::failure& e) {
			std::cerr << e.what() << std::endl;
			exit(-3);
		}
	}
	return i;
}

void  ndt_mapping::parseFilteredScan(std::ifstream& input_file, PointCloud& pointcloud) {
	int32_t size;
	try {
		input_file.read((char*)&size, sizeof(int32_t));
		pointcloud.clear();
		for (int i = 0; i < size; i++)
		{
			PointXYZI p;
			input_file.read((char*)&p.data[0], sizeof(float));
			input_file.read((char*)&p.data[1], sizeof(float));
			input_file.read((char*)&p.data[2], sizeof(float));
			input_file.read((char*)&p.data[3], sizeof(float));
			pointcloud.push_back(p);
		}
	}  catch (std::ifstream::failure& e) {
		throw std::ios_base::failure("Error reading filtered scan");
	}
}


void ndt_mapping::parseInitGuess(std::ifstream& input_file, Matrix4f& initGuess) {
	try {
		for (int h = 0; h < 4; h++)
			for (int w = 0; w < 4; w++)
				input_file.read((char*)&(initGuess.data[h][w]),sizeof(float));
	}  catch (std::ifstream::failure& e) {
		throw std::ios_base::failure("Error reading initial guess");
	}
}

/**
 * Reads the next reference matrix.
 */
void ndt_mapping::parseResult(std::ifstream& output_file, CallbackResult& result) {
	try {
		for (int h = 0; h < 4; h++)
			for (int w = 0; w < 4; w++)
			{
				float m;
				output_file.read((char*)&m, sizeof(float));
				result.final_transformation.data[h][w] = m;
			}
		double fitness;
		output_file.read((char*)&fitness, sizeof(double));
		result.fitness_score = fitness;
		bool converged;
		output_file.read((char*)&converged, sizeof(bool));
		result.converged = converged;
	}  catch (std::ifstream::failure& e) {
		throw std::ios_base::failure("Error reading result.");
	}
}

void ndt_mapping::parseIntermediateResults(std::ifstream& output_file, CallbackResult& result) {

	try {
		int resultNo;
		output_file.read((char*)&resultNo, sizeof(int32_t));
		result.intermediate_transformations.resize(resultNo);
		for (int i = 0; i < resultNo; i++) {
			Matrix4f& m = result.intermediate_transformations[i];
			for (int h = 0; h < 4; h++) {
				for (int w = 0; w < 4; w++) {
					output_file.read((char*)&(m.data[h][w]), sizeof(float));
				}
			}
		}
	} catch (std::ifstream::failure& e) {
		throw std::ios_base::failure("Error reading voxel grid.");
	}
}
#ifdef EPHOS_DATAGEN
void ndt_mapping::writeResult(std::ofstream& output_file, CallbackResult& result) {
	try {
		Matrix4f& m = result.final_transformation;
		for (int h = 0; h < 4; h++) {
			for (int w = 0; w < 4; w++) {
				output_file.write((char*)&(m.data[h][w]), sizeof(float));
			}
		}
		double fitness = result.fitness_score;
		output_file.write((char*)&fitness, sizeof(double));
		bool converged = result.converged;
		output_file.write((char*)&converged, sizeof(bool));
	} catch (std::ofstream::failure& e) {
		throw std::ios_base::failure("Error writeing result.");
	}
}
void ndt_mapping::writeIntermediateResults(std::ofstream& output_file, CallbackResult& result) {

	try {
		int resultNo = result.intermediate_transformations.size();
		output_file.write((char*)&resultNo, sizeof(int32_t));
		for (int i = 0; i < resultNo; i++) {
			Matrix4f& m = result.intermediate_transformations[i];
			for (int h = 0; h < 4; h++) {
				for (int w = 0; w < 4; w++) {
					output_file.write((char*)&(m.data[h][w]), sizeof(float));
				}
			}
		}
	} catch (std::ofstream::failure& e) {
		throw std::ios_base::failure("Error writing voxel grid. ");
	}
}
#endif // EPHOS_DATAGEN
int ndt_mapping::read_number_testcases(std::ifstream& input_file)
{
	int32_t caseNo;
	try {
		input_file.read((char*)&caseNo, sizeof(int32_t));
	} catch (std::ifstream::failure) {
		throw std::ios_base::failure("Error reading number of test cases");
	}
	return caseNo;
}

void ndt_mapping::run(int p) {
	std::cout << "executing for " << testcases << " test cases" << std::endl;
	start_timer();
	pause_timer();
	while (read_testcases < testcases)
	{
		int count = read_next_testcases(p);

		resume_timer();
		for (int i = 0; i < count; i++)
		{
			// actual kernel invocation
			results[i] = partial_points_callback(
				filtered_scan[i],
				init_guess[i],
				maps[i]
			);
		}
		pause_timer();
		check_next_outputs(count);
	}
	stop_timer();
}

void ndt_mapping::check_next_outputs(int count)
{
	CallbackResult reference;
	for (int i = 0; i < count; i++)
	{
		try {
			parseResult(output_file, reference);
			parseIntermediateResults(output_file, reference);
#ifdef EPHOS_DATAGEN
			writeResult(datagen_file, results[i]);
			writeIntermediateResults(datagen_file, results[i]);
#endif
		} catch (std::ios_base::failure& e) {
			std::cerr << e.what() << std::endl;
			exit(-3);
		}
		if (results[i].converged != reference.converged)
		{
			error_so_far = true;
		}
		// compare the matrices
		for (int h = 0; h < 4; h++) {
			// test for nan
			for (int w = 0; w < 4; w++) {
				if (std::isnan(results[i].final_transformation.data[h][w]) !=
					std::isnan(reference.final_transformation.data[h][w])) {
					error_so_far = true;
				}
			}
			// compare translation
			float delta = std::fabs(results[i].final_transformation.data[h][3] -
				reference.final_transformation.data[h][3]);
			if (delta > max_delta) {
				max_delta = delta;
				if (delta > MAX_TRANSLATION_EPS) {
					error_so_far = true;
				}
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
				if (delta > MAX_EPS) {
					error_so_far = true;
				}
			}
		}
		std::ostringstream sError;
		int caseErrorNo = 0;
		for (int w = 0; w < 4; w++) {
			float delta = std::fabs(resPoint.data[w] - refPoint.data[w]);
			if (delta > max_delta) {
				max_delta = delta;
				if (delta > MAX_ROTATION_EPS) {
					error_so_far = true;
				}
			}
			if (delta > MAX_ROTATION_EPS) {
				sError << " mismatch vector[" << w << "]: ";
				sError << refPoint.data[w] << " should be " << resPoint.data[w] << std::endl;
				caseErrorNo += 1;
			}
		}
		if (caseErrorNo > 0) {
			std::cout << "Errors for test case " << read_testcases - count + i;
			std::cout << sError.str() << std::endl;
		}
	}
}

bool ndt_mapping::check_output() {
	std::cout << "checking output \n";
	// check for error
	std::cout << "max delta: " << max_delta << "\n";
	return !error_so_far;
}

// set the kernel used in main()
ndt_mapping a = ndt_mapping();
benchmark& myKernel = a;