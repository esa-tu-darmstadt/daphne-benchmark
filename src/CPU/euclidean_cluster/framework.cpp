
/**
 * Author:  Florian Stock, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2018
 * License: Apache 2.0 (see attachached File)
 */
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>

#include "kernel.h"
#include "benchmark.h"
#include "datatypes.h"

/**
 * Reads the next point cloud.
 */
void parsePointCloud(std::ifstream& input_file, PointCloud *cloud)
{
	int size = 0;
	Point p;
	input_file.read((char*)&(size), sizeof(int));
	try {
		for (int i = 0; i < size; i++)
		{
			input_file.read((char*)&p.x, sizeof(float));
			input_file.read((char*)&p.y, sizeof(float));
			input_file.read((char*)&p.z, sizeof(float));
			cloud->push_back(p);
		}
	} catch (std::ifstream::failure) {
		throw std::ios_base::failure("Error reading point cloud");
	}
}

/**
 * Reads the next reference cloud result.
 */
void parseOutCloud(std::ifstream& input_file, PointCloudRGB *cloud)
{
    int size = 0;
    PointRGB p;
    try {
	input_file.read((char*)&(size), sizeof(int));

	for (int i = 0; i < size; i++)
	    {
		input_file.read((char*)&p.x, sizeof(float));
		input_file.read((char*)&p.y, sizeof(float));
		input_file.read((char*)&p.z, sizeof(float));
		input_file.read((char*)&p.r, sizeof(uint8_t));
		input_file.read((char*)&p.g, sizeof(uint8_t));
		input_file.read((char*)&p.b, sizeof(uint8_t));
		cloud->push_back(p);
	    }
    }  catch (std::ifstream::failure) {
		throw std::ios_base::failure("Error reading reference cloud");
    }
}

/**
 * Reads the next reference bounding boxes.
 */
void parseBoundingboxArray(std::ifstream& input_file, BoundingboxArray *bb_array)
{
    int size = 0;
    Boundingbox bba;
    try {
		input_file.read((char*)&(size), sizeof(int));
		for (int i = 0; i < size; i++)
		{
			input_file.read((char*)&bba.position.x, sizeof(double));
			input_file.read((char*)&bba.position.y, sizeof(double));
			input_file.read((char*)&bba.orientation.x, sizeof(double));
			input_file.read((char*)&bba.orientation.y, sizeof(double));
			input_file.read((char*)&bba.orientation.z, sizeof(double));
			input_file.read((char*)&bba.orientation.w, sizeof(double));
			input_file.read((char*)&bba.dimensions.x, sizeof(double));
			input_file.read((char*)&bba.dimensions.y, sizeof(double));
			bb_array->boxes.push_back(bba);
	    }
    }  catch (std::ifstream::failure) {
		throw std::ios_base::failure("Error reading reference bounding boxes");
    }
}

/*
 * Reads the next reference centroids.
 */
void parseCentroids(std::ifstream& input_file, Centroid *centroids)
{
	int size = 0;
	PointDouble p;
	try {
		input_file.read((char*)&(size), sizeof(int));
		for (int i = 0; i < size; i++)
		{
			input_file.read((char*)&p.x, sizeof(double));
			input_file.read((char*)&p.y, sizeof(double));
			input_file.read((char*)&p.z, sizeof(double));
			centroids->points.push_back(p);
		}
	} catch (std::ifstream::failure) {
		throw std::ios_base::failure("Error reading reference centroids");
	}
}
int euclidean_clustering::read_number_testcases(std::ifstream& input_file)
{
	int32_t number;
	try {
		input_file.read((char*)&(number), sizeof(int32_t));
	} catch (std::ifstream::failure) {
		throw std::ios_base::failure("Error reading the number of testcases");
	}
	return number;
}
int euclidean_clustering::read_next_testcases(int count)
{
	int i;
	// free previously allocated memory
	delete [] in_cloud_ptr;
	delete [] out_cloud_ptr;
	delete [] out_boundingbox_array;
	delete [] out_centroids;
	// allocate new memory for the current case
	in_cloud_ptr = new PointCloud[count];
	out_cloud_ptr = new PointCloudRGB[count];
	out_boundingbox_array = new BoundingboxArray[count];
	out_centroids = new Centroid[count];
	// read the testcase data
	for (i = 0; (i < count) && (read_testcases < testcases); i++,read_testcases++)
	{
		try {
			parsePointCloud(input_file, in_cloud_ptr + i);
		} catch (std::ios_base::failure& e) {
			std::cerr << e.what() << std::endl;
			exit(-3);
		}
	}
	return i;
}

void euclidean_clustering::check_next_outputs(int count)
{
	PointCloudRGB reference_out_cloud;
	BoundingboxArray reference_bb_array;
	Centroid reference_centroids;

	for (int i = 0; i < count; i++)
	{
		// read the reference result
		try {
			parseOutCloud(output_file, &reference_out_cloud);
			parseBoundingboxArray(output_file, &reference_bb_array);
			parseCentroids(output_file, &reference_centroids);
		} catch (std::ios_base::failure& e) {
			std::cerr << e.what() << std::endl;
			exit(-3);
		}

		// as the result is still right when points/boxes/centroids are in different order,
		// we sort the result and reference to normalize it and we can compare it
		std::sort(reference_out_cloud.begin(), reference_out_cloud.end(), compareRGBPoints);
		std::sort(out_cloud_ptr[i].begin(), out_cloud_ptr[i].end(), compareRGBPoints);
		std::sort(reference_bb_array.boxes.begin(), reference_bb_array.boxes.end(), compareBBs);
		std::sort(out_boundingbox_array[i].boxes.begin(), out_boundingbox_array[i].boxes.end(), compareBBs);
		std::sort(reference_centroids.points.begin(), reference_centroids.points.end(), comparePoints);
		std::sort(out_centroids[i].points.begin(), out_centroids[i].points.end(), comparePoints);
		// test for size differences
		std::ostringstream sError;
		int caseErrorNo = 0;
		// test for size differences
		if (reference_out_cloud.size() != out_cloud_ptr[i].size())
		{
			error_so_far = true;
			caseErrorNo += 1;
			//sError << " invalid point number: " << out_cloud_ptr[i].size();
			//sError << " should be " << reference_out_cloud.size() << std::endl;
		}
		if (reference_bb_array.boxes.size() != out_boundingbox_array[i].boxes.size())
		{
			error_so_far = true;
			caseErrorNo += 1;
			//sError << " invalid bounding box number: " << out_boundingbox_array[i].boxes.size();
			//sError << " should be " << reference_bb_array.boxes.size() << std::endl;
		}
		if (reference_centroids.points.size() != out_centroids[i].points.size())
		{
			error_so_far = true;
			caseErrorNo += 1;
			//sError << " invalid centroid number: " << out_centroids[i].points.size();
			//sError << " should be " << reference_centroids.points.size() << std::endl;
		}
		if (caseErrorNo == 0) {
			// test for content divergence
			for (int j = 0; j < reference_out_cloud.size(); j++)
			{
				float deltaX = std::abs(out_cloud_ptr[i][j].x - reference_out_cloud[j].x);
				float deltaY = std::abs(out_cloud_ptr[i][j].y - reference_out_cloud[j].y);
				float deltaZ = std::abs(out_cloud_ptr[i][j].z - reference_out_cloud[j].z);
				float delta = std::fmax(deltaX, std::fmax(deltaY, deltaZ));
				if (delta > MAX_EPS) {
					caseErrorNo += 1;
// 					sError << " deviating point " << j << ": (";
// 					sError << out_cloud_ptr[i][j].x << " " << out_cloud_ptr[i][j].y << " " << out_cloud_ptr[i][j].z;
// 					sError << ") should be (" << reference_out_cloud[j].x << " ";
// 					sError << reference_out_cloud[j].y << " " << reference_out_cloud[j].z << ")" << std::endl;
					if (delta > max_delta) {
						max_delta = delta;
					}
				}
				//max_delta = std::fmax(std::abs(out_cloud_ptr[i][j].x - reference_out_cloud[j].x), max_delta);
				//max_delta = std::fmax(std::abs(out_cloud_ptr[i][j].y - reference_out_cloud[j].y), max_delta);
				//max_delta = std::fmax(std::abs(out_cloud_ptr[i][j].z - reference_out_cloud[j].z), max_delta);
			}
			for (int j = 0; j < reference_bb_array.boxes.size(); j++)
			{
				float deltaX = std::abs(out_boundingbox_array[i].boxes[j].position.x - reference_bb_array.boxes[j].position.x);
				float deltaY = std::abs(out_boundingbox_array[i].boxes[j].position.y - reference_bb_array.boxes[j].position.y);
				float deltaW = std::abs(out_boundingbox_array[i].boxes[j].dimensions.x - reference_bb_array.boxes[j].dimensions.x);
				float deltaH = std::abs(out_boundingbox_array[i].boxes[j].dimensions.y - reference_bb_array.boxes[j].dimensions.y);
				float deltaOX = std::abs(out_boundingbox_array[i].boxes[j].orientation.x - reference_bb_array.boxes[j].orientation.x);
				float deltaOY = std::abs(out_boundingbox_array[i].boxes[j].orientation.y - reference_bb_array.boxes[j].orientation.y);
				float deltaP = std::fmax(deltaX, deltaY);
				float deltaS = std::fmax(deltaW, deltaH);
				float deltaO = std::fmax(deltaOX, deltaOY);
				float delta = 0;
				if (deltaP > MAX_EPS) {
					delta = std::fmax(delta, deltaP);
// 					sError << " deviating bounding box " << j << " position: (";
// 					sError << out_boundingbox_array[i].boxes[j].position.x << " ";
// 					sError << out_boundingbox_array[i].boxes[j].position.y << ") should be (";
// 					sError << reference_bb_array.boxes[j].position.x << " ";
// 					sError << reference_bb_array.boxes[j].position.y << ")" << std::endl;
				}
				if (deltaS > MAX_EPS) {
					delta = std::fmax(delta, deltaS);
// 					sError << " deviating bounding box " << j << " size: (";
// 					sError << out_boundingbox_array[i].boxes[j].dimensions.x << " ";
// 					sError << out_boundingbox_array[i].boxes[j].dimensions.y << ") should be (";
// 					sError << reference_bb_array.boxes[j].dimensions.x << " ";
// 					sError << reference_bb_array.boxes[j].dimensions.y << ")" << std::endl;
				}
				if (deltaO > MAX_EPS) {
					delta = std::fmax(delta, deltaO);
// 					sError << " deviating bound box " << j << " orientation: (";
// 					sError << out_boundingbox_array[i].boxes[j].orientation.x << " ";
// 					sError << out_boundingbox_array[i].boxes[j].orientation.y << ") should be (";
// 					sError << reference_bb_array.boxes[j].orientation.x << " ";
// 					sError << reference_bb_array.boxes[j].orientation.y << ")" << std::endl;
				}
				if (delta > MAX_EPS) {
					caseErrorNo += 1;
					if (delta > max_delta) {
						max_delta = delta;
					}
				}
			}
			for (int j = 0; j < reference_centroids.points.size(); j++)
			{
				float deltaX = std::abs(out_centroids[i].points[j].x - reference_centroids.points[j].x);
				float deltaY = std::abs(out_centroids[i].points[j].y - reference_centroids.points[j].y);
				float deltaZ = std::abs(out_centroids[i].points[j].z - reference_centroids.points[j].z);
				float delta = std::fmax(deltaX, std::fmax(deltaY, deltaZ));
				if (delta > MAX_EPS) {
					caseErrorNo += 1;
					if (delta > max_delta) {
						max_delta = delta;
					}
// 					sError << " deviating centroid " << j << " position: (";
// 					sError << out_centroids[i].points[j].x << " " << out_centroids[i].points[j].y << " ";
// 					sError << out_centroids[i].points[j].z << ") should be (";
// 					sError << reference_centroids.points[j].x << " " << reference_centroids.points[j].y << " ";
// 					sError << reference_centroids.points[j].z << ")" << std::endl;
				}
			}
		}
		if (caseErrorNo > 0) {
			std::cerr << "Errors for test case " << read_testcases - count + i;
			std::cerr << " (" << caseErrorNo << "):" << std::endl;
			std::cerr << sError.str() << std::endl;
		}
		// finishing steps for the next iteration
		reference_bb_array.boxes.clear();
		reference_out_cloud.clear();
		reference_centroids.points.clear();
	}
}
bool euclidean_clustering::check_output()
{
	std::cout << "checking output \n";

	// acts as complement to init()
	input_file.close();
	output_file.close();
	std::cout << "max delta: " << max_delta << "\n";
	if ((max_delta > MAX_EPS) || error_so_far)
	{
		return false;
	} else
	{
		return true;
	}
}

// set kernel used by main()
euclidean_clustering a = euclidean_clustering();
kernel& myKernel = a;
