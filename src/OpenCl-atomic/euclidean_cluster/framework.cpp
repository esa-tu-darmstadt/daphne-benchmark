/**
 * Author:  Florian Stock, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2018
 * Author:  Leonardo Solis, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2018
 * Author:  Thilo Gabel, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2019
 * License: Apache 2.0 (see attachached File)
 */
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>
#include <cstring>

#include "common/benchmark.h"
#include "datatypes.h"
#include "euclidean_clustering.h"

void euclidean_clustering::parsePointCloud(std::ifstream& input_file, PointCloud *cloud, int *cloudSize)
{
	input_file.read((char*)(cloudSize), sizeof(int));
	*cloud = (Point*) malloc(sizeof(Point) * (*cloudSize));
	try {
	for (int i = 0; i < *cloudSize; i++)
		{
		input_file.read((char*)&(*cloud)[i].x, sizeof(float));
		input_file.read((char*)&(*cloud)[i].y, sizeof(float));
		input_file.read((char*)&(*cloud)[i].z, sizeof(float));
		}
	} catch (std::ifstream::failure e) {
		throw std::ios_base::failure("Error reading point cloud");
	}
}
/**
 * Reads the next reference cloud result.
 */
void euclidean_clustering::parseOutCloud(std::ifstream& input_file, PointCloudRGB *cloud)
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


void euclidean_clustering::parseBoundingboxArray(std::ifstream& input_file, BoundingboxArray *bb_array)
{
	int size = 0;
	Boundingbox bba;
	#if defined (DOUBLE_FP)
	#else
	double temp;
	#endif
	try {
		input_file.read((char*)&(size), sizeof(int));
		for (int i = 0; i < size; i++)
		{
			#if defined (DOUBLE_FP)
			input_file.read((char*)&bba.position.x, sizeof(double));
			input_file.read((char*)&bba.position.y, sizeof(double));
			input_file.read((char*)&bba.orientation.x, sizeof(double));
			input_file.read((char*)&bba.orientation.y, sizeof(double));
			input_file.read((char*)&bba.orientation.z, sizeof(double));
			input_file.read((char*)&bba.orientation.w, sizeof(double));
			input_file.read((char*)&bba.dimensions.x, sizeof(double));
			input_file.read((char*)&bba.dimensions.y, sizeof(double));
			#else
			input_file.read((char*)&temp, sizeof(double));
			bba.position.x=temp;
			input_file.read((char*)&temp, sizeof(double));
			bba.position.y=temp;
			input_file.read((char*)&temp, sizeof(double));
			bba.orientation.x=temp;
			input_file.read((char*)&temp, sizeof(double));
			bba.orientation.y=temp;
			input_file.read((char*)&temp, sizeof(double));
			bba.orientation.z=temp;
			input_file.read((char*)&temp, sizeof(double));
			bba.orientation.w=temp;
			input_file.read((char*)&temp, sizeof(double));
			bba.dimensions.x=temp;
			input_file.read((char*)&temp, sizeof(double));
			bba.dimensions.y=temp;
			#endif
			bb_array->boxes.push_back(bba);
		}
	}  catch (std::ifstream::failure e) {
		throw std::ios_base::failure("Error reading reference bounding boxes");
	}
}

/*
 * Reads the next reference centroids.
 */
void euclidean_clustering::parseCentroids(std::ifstream& input_file, Centroid *centroids)
{
	int size = 0;
	PointDouble p;
	#if defined (DOUBLE_FP)
	#else
	double temp;
	#endif
	try {
	input_file.read((char*)&(size), sizeof(int));
		for (int i = 0; i < size; i++)
		{
			#if defined (DOUBLE_FP)
			input_file.read((char*)&p.x, sizeof(double));
			input_file.read((char*)&p.y, sizeof(double));
			input_file.read((char*)&p.z, sizeof(double));
			#else
			input_file.read((char*)&temp, sizeof(double));
			p.x = temp;
			input_file.read((char*)&temp, sizeof(double));
			p.y = temp;
			input_file.read((char*)&temp, sizeof(double));
			p.z = temp;
			#endif
			centroids->points.push_back(p);
		}
    } catch (std::ifstream::failure e) {
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
	// free memory of the last iteration and allocate new one
	int i;
	delete [] in_cloud_ptr;
	delete [] cloud_size;
	delete [] out_cloud_ptr;
	delete [] out_boundingbox_array;
	delete [] out_centroids;
	in_cloud_ptr = new PointCloud[count];
	cloud_size = new int [count];
	out_cloud_ptr = new PointCloudRGB[count];
	out_boundingbox_array = new BoundingboxArray[count];
	out_centroids = new Centroid[count];
	// read the respective point clouds
	for (i = 0; (i < count) && (read_testcases < testcases); i++,read_testcases++)
	{
		try {
			parsePointCloud(input_file, &in_cloud_ptr[i], &cloud_size[i]);
		} catch (std::ios_base::failure& e) {
			std::cerr << e.what() << std::endl;
			exit(-3);
		}
	}
	return i;
}


/**
 * Helper function for point comparison
 */
inline bool compareRGBPoints (const PointRGB &a, const PointRGB &b)
{
    if (a.x != b.x)
		return (a.x < b.x);
    else
	if (a.y != b.y)
	    return (a.y < b.y);
	else
	    return (a.z < b.z);
}

/**
 * Helper function for point comparison
 */
inline bool comparePoints (const PointDouble &a, const PointDouble &b)
{
	if (a.x != b.x)
		return (a.x < b.x);
	else
	if (a.y != b.y)
		return (a.y < b.y);
	else
		return (a.z < b.z);
}


/**
 * Helper function for bounding box comparison
 */
inline bool compareBBs (const Boundingbox &a, const Boundingbox &b)
{
	if (a.position.x != b.position.x)
		return (a.position.x < b.position.x);
	else
	if (a.position.y != b.position.y)
		return (a.position.y < b.position.y);
	else
		if (a.dimensions.x != b.dimensions.x)
			return (a.dimensions.x < b.dimensions.x);
		else
			return (a.dimensions.y < b.dimensions.y);
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
		if (reference_out_cloud.size() != out_cloud_ptr[i].size())
		{
			error_so_far = true;
			continue;
		}
		if (reference_bb_array.boxes.size() != out_boundingbox_array[i].boxes.size())
		{
			error_so_far = true;
			continue;
		}
		if (reference_centroids.points.size() != out_centroids[i].points.size())
		{
			error_so_far = true;
			continue;
		}
		// test for content divergence
		for (int j = 0; j < reference_out_cloud.size(); j++)
		{
			max_delta = std::fmax(std::abs(out_cloud_ptr[i][j].x - reference_out_cloud[j].x), max_delta);
			max_delta = std::fmax(std::abs(out_cloud_ptr[i][j].y - reference_out_cloud[j].y), max_delta);
			max_delta = std::fmax(std::abs(out_cloud_ptr[i][j].z - reference_out_cloud[j].z), max_delta);
		}
		for (int j = 0; j < reference_bb_array.boxes.size(); j++)
		{
			max_delta = std::fmax(std::abs(out_boundingbox_array[i].boxes[j].position.x - reference_bb_array.boxes[j].position.x), max_delta);
			max_delta = std::fmax(std::abs(out_boundingbox_array[i].boxes[j].position.y - reference_bb_array.boxes[j].position.y), max_delta);
			max_delta = std::fmax(std::abs(out_boundingbox_array[i].boxes[j].dimensions.x - reference_bb_array.boxes[j].dimensions.x), max_delta);
			max_delta = std::fmax(std::abs(out_boundingbox_array[i].boxes[j].dimensions.y - reference_bb_array.boxes[j].dimensions.y), max_delta);
			max_delta = std::fmax(std::abs(out_boundingbox_array[i].boxes[j].orientation.x - reference_bb_array.boxes[j].orientation.x), max_delta);
			max_delta = std::fmax(std::abs(out_boundingbox_array[i].boxes[j].orientation.y - reference_bb_array.boxes[j].orientation.y), max_delta);
		}
		for (int j = 0; j < reference_centroids.points.size(); j++)
		{
			max_delta = std::fmax(std::abs(out_centroids[i].points[j].x - reference_centroids.points[j].x), max_delta);
			max_delta = std::fmax(std::abs(out_centroids[i].points[j].y - reference_centroids.points[j].y), max_delta);
			max_delta = std::fmax(std::abs(out_centroids[i].points[j].z - reference_centroids.points[j].z), max_delta);
		}
		// finishing steps for the next iteration
		reference_bb_array.boxes.clear();
		reference_out_cloud.clear();
		reference_centroids.points.clear();
	}
}

void euclidean_clustering::run(int p) {
	start_timer();
	pause_timer();
	while (read_testcases < testcases)
	{
		// read the next input data
		int count = read_next_testcases(p);
		resume_timer();
		for (int i = 0; i < count; i++)
		{
			// actual kernel invocation
			segmentByDistance(
				in_cloud_ptr[i],
				cloud_size[i],
				&out_cloud_ptr[i],
				&out_boundingbox_array[i],
				&out_centroids[i]
			);
		}
		// pause the timer, then read and compare with the reference data
		pause_timer();
		check_next_outputs(count);
	}
	stop_timer();
}

bool euclidean_clustering::check_output()
{
	std::cout << "checking output \n";

	// acts as complement to init()

	std::cout << "max delta: " << max_delta << "\n";
	if ((max_delta > MAX_EPS) || error_so_far)
	{
		return false;
	} else
	{
		return true;
	}
}


euclidean_clustering a;
benchmark& myKernel = a;
