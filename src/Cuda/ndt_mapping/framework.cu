/**
 * Author:  Florian Stock, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2018
 * Author:  Thilo Gabel, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2019 - 2020
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
			parseMaps(input_file, maps[i]);
		} catch (std::ios_base::failure& e) {
			std::cerr << e.what() << std::endl;
			exit(-3);
		}
	}
	return i;
}
void ndt_mapping::cleanupTestcases(int count) {
	// free memory allocated by parsers
	for (int i = 0; i < count; i++) {
		delete[] filtered_scan[i].data;
	}
	filtered_scan.resize(0);
	for (int i = 0; i < count; i++) {
		cudaFree(maps[i].data);
	}
	maps.resize(0);
	init_guess.resize(0);
	results.resize(0);
}
void  ndt_mapping::parseMaps(std::ifstream& input_file, PointCloud& pointcloud) {
	int32_t cloudSize;
	try {
		input_file.read((char*)&cloudSize, sizeof(int32_t));
		// TODO make sure to not create a memory leak here
		//pointcloud.clear();
		cudaMallocManaged(&pointcloud.data, sizeof(PointXYZI)*cloudSize);
		pointcloud.size = cloudSize;
		pointcloud.capacity = cloudSize;
		for (int i = 0; i < cloudSize; i++)
		{
			PointXYZI p;
			input_file.read((char*)&p.data[0], sizeof(float));
			input_file.read((char*)&p.data[1], sizeof(float));
			input_file.read((char*)&p.data[2], sizeof(float));
			input_file.read((char*)&p.data[3], sizeof(float));
			pointcloud.data[i] = p;
		}
	}  catch (std::ifstream::failure& e) {
		throw std::ios_base::failure("Error reading filtered scan");
	}
}