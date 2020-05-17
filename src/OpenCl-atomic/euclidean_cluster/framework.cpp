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

void euclidean_clustering::parsePlainPointCloud(std::ifstream& input_file, PlainPointCloud& cloud,
	int& cloudSize)
{
	input_file.read((char*)&cloudSize, sizeof(int));
	//*cloud = (Point*) malloc(sizeof(Point) * (*cloudSize));
	cloud = new Point[cloudSize];
	try {
	for (int i = 0; i < cloudSize; i++)
		{
		input_file.read((char*)&(cloud[i].x), sizeof(float));
		input_file.read((char*)&(cloud[i].y), sizeof(float));
		input_file.read((char*)&(cloud[i].z), sizeof(float));
		}
	} catch (std::ifstream::failure e) {
		throw std::ios_base::failure("Error reading point cloud");
	}
}
/**
 * Reads the next reference cloud result.
 */
void euclidean_clustering::parseColorPointCloud(std::ifstream& input_file, ColorPointCloud& cloud)
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
		cloud.push_back(p);
	    }
    }  catch (std::ifstream::failure) {
		throw std::ios_base::failure("Error reading reference cloud");
    }
}


void euclidean_clustering::parseBoundingboxArray(std::ifstream& input_file, BoundingboxArray& bb_array)
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
			bb_array.boxes.push_back(bba);
		}
	}  catch (std::ifstream::failure e) {
		throw std::ios_base::failure("Error reading reference bounding boxes");
	}
}

/*
 * Reads the next reference centroids.
 */
void euclidean_clustering::parseCentroids(std::ifstream& input_file, Centroid& centroids)
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
			centroids.points.push_back(p);
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
	plainPointCloud.resize(count);
	colorPointCloud.resize(count);
	clusterBoundingBoxes.resize(count);
	clusterCentroids.resize(count);
	plainCloudSize.resize(count);

	// read the respective point clouds
	for (i = 0; (i < count) && (read_testcases < testcases); i++,read_testcases++)
	{
		try {
			parsePlainPointCloud(input_file, plainPointCloud[i], plainCloudSize[i]);
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
	ColorPointCloud refPointCloud;
	BoundingboxArray refBoundingBoxes;
	Centroid refClusterCentroids;

	for (int i = 0; i < count; i++)
	{
		// read the reference result
		try {
			parseColorPointCloud(output_file, refPointCloud);
			parseBoundingboxArray(output_file, refBoundingBoxes);
			parseCentroids(output_file, refClusterCentroids);
		} catch (std::ios_base::failure& e) {
			std::cerr << e.what() << std::endl;
			exit(-3);
		}

		// as the result is still right when points/boxes/centroids are in different order,
		// we sort the result and reference to normalize it and we can compare it
		std::sort(refPointCloud.begin(), refPointCloud.end(), compareRGBPoints);
		std::sort(colorPointCloud[i].begin(), colorPointCloud[i].end(), compareRGBPoints);
		std::sort(refBoundingBoxes.boxes.begin(), refBoundingBoxes.boxes.end(), compareBBs);
		std::sort(clusterBoundingBoxes[i].boxes.begin(), clusterBoundingBoxes[i].boxes.end(), compareBBs);
		std::sort(refClusterCentroids.points.begin(), refClusterCentroids.points.end(), comparePoints);
		std::sort(clusterCentroids[i].points.begin(), clusterCentroids[i].points.end(), comparePoints);
		// test for size differences
		std::ostringstream sError;
		int caseErrorNo = 0;
		// test for size differences
		if (refPointCloud.size() != colorPointCloud[i].size())
		{
			error_so_far = true;
			caseErrorNo += 1;
			sError << " invalid point number: " << colorPointCloud[i].size();
			sError << " should be " << refPointCloud.size() << std::endl;
		}
		if (refBoundingBoxes.boxes.size() != clusterBoundingBoxes[i].boxes.size())
		{
			error_so_far = true;
			caseErrorNo += 1;
			sError << " invalid bounding box number: " << clusterBoundingBoxes[i].boxes.size();
			sError << " should be " << refBoundingBoxes.boxes.size() << std::endl;
		}
		if (refClusterCentroids.points.size() != clusterCentroids[i].points.size())
		{
			error_so_far = true;
			caseErrorNo += 1;
			sError << " invalid centroid number: " << clusterCentroids[i].points.size();
			sError << " should be " << refClusterCentroids.points.size() << std::endl;
		}
		if (caseErrorNo == 0) {
			// test for content divergence
			for (int j = 0; j < refPointCloud.size(); j++)
			{
				float deltaX = std::abs(colorPointCloud[i][j].x - refPointCloud[j].x);
				float deltaY = std::abs(colorPointCloud[i][j].y - refPointCloud[j].y);
				float deltaZ = std::abs(colorPointCloud[i][j].z - refPointCloud[j].z);
				float delta = std::fmax(deltaX, std::fmax(deltaY, deltaZ));
				if (delta > MAX_EPS) {
					caseErrorNo += 1;
					sError << " deviating point " << j << ": (";
					sError << colorPointCloud[i][j].x << " " << colorPointCloud[i][j].y << " " << colorPointCloud[i][j].z;
					sError << ") should be (" << refPointCloud[j].x << " ";
					sError << refPointCloud[j].y << " " << refPointCloud[j].z << ")" << std::endl;
					if (delta > max_delta) {
						max_delta = delta;
					}
				}
			}
			for (int j = 0; j < refBoundingBoxes.boxes.size(); j++)
			{
				float deltaX = std::abs(clusterBoundingBoxes[i].boxes[j].position.x - refBoundingBoxes.boxes[j].position.x);
				float deltaY = std::abs(clusterBoundingBoxes[i].boxes[j].position.y - refBoundingBoxes.boxes[j].position.y);
				float deltaW = std::abs(clusterBoundingBoxes[i].boxes[j].dimensions.x - refBoundingBoxes.boxes[j].dimensions.x);
				float deltaH = std::abs(clusterBoundingBoxes[i].boxes[j].dimensions.y - refBoundingBoxes.boxes[j].dimensions.y);
				float deltaOX = std::abs(clusterBoundingBoxes[i].boxes[j].orientation.x - refBoundingBoxes.boxes[j].orientation.x);
				float deltaOY = std::abs(clusterBoundingBoxes[i].boxes[j].orientation.y - refBoundingBoxes.boxes[j].orientation.y);
				float deltaP = std::fmax(deltaX, deltaY);
				float deltaS = std::fmax(deltaW, deltaH);
				float deltaO = std::fmax(deltaOX, deltaOY);
				float delta = 0;
				if (deltaP > MAX_EPS) {
					delta = std::fmax(delta, deltaP);
					sError << " deviating bounding box " << j << " position: (";
					sError << clusterBoundingBoxes[i].boxes[j].position.x << " ";
					sError << clusterBoundingBoxes[i].boxes[j].position.y << ") should be (";
					sError << refBoundingBoxes.boxes[j].position.x << " ";
					sError << refBoundingBoxes.boxes[j].position.y << ")" << std::endl;
				}
				if (deltaS > MAX_EPS) {
					delta = std::fmax(delta, deltaS);
					sError << " deviating bounding box " << j << " size: (";
					sError << clusterBoundingBoxes[i].boxes[j].dimensions.x << " ";
					sError << clusterBoundingBoxes[i].boxes[j].dimensions.y << ") should be (";
					sError << refBoundingBoxes.boxes[j].dimensions.x << " ";
					sError << refBoundingBoxes.boxes[j].dimensions.y << ")" << std::endl;
				}
				if (deltaO > MAX_EPS) {
					delta = std::fmax(delta, deltaO);
					sError << " deviating bound box " << j << " orientation: (";
					sError << clusterBoundingBoxes[i].boxes[j].orientation.x << " ";
					sError << clusterBoundingBoxes[i].boxes[j].orientation.y << ") should be (";
					sError << refBoundingBoxes.boxes[j].orientation.x << " ";
					sError << refBoundingBoxes.boxes[j].orientation.y << ")" << std::endl;
				}
				if (delta > MAX_EPS) {
					caseErrorNo += 1;
					if (delta > max_delta) {
						max_delta = delta;
					}
				}
			}
			for (int j = 0; j < refClusterCentroids.points.size(); j++)
			{
				float deltaX = std::abs(clusterCentroids[i].points[j].x - refClusterCentroids.points[j].x);
				float deltaY = std::abs(clusterCentroids[i].points[j].y - refClusterCentroids.points[j].y);
				float deltaZ = std::abs(clusterCentroids[i].points[j].z - refClusterCentroids.points[j].z);
				float delta = std::fmax(deltaX, std::fmax(deltaY, deltaZ));
				if (delta > MAX_EPS) {
					caseErrorNo += 1;
					if (delta > max_delta) {
						max_delta = delta;
					}
					sError << " deviating centroid " << j << " position: (";
					sError << clusterCentroids[i].points[j].x << " " << clusterCentroids[i].points[j].y << " ";
					sError << clusterCentroids[i].points[j].z << ") should be (";
					sError << refClusterCentroids.points[j].x << " " << refClusterCentroids.points[j].y << " ";
					sError << refClusterCentroids.points[j].z << ")" << std::endl;
				}
			}
		}
		if (caseErrorNo > 0) {
			std::cerr << "Errors for test case " << read_testcases - count + i;
			std::cerr << " (" << caseErrorNo << "):" << std::endl;
			std::cerr << sError.str() << std::endl;
		}
		// finishing steps for the next iteration
		refBoundingBoxes.boxes.clear();
		refPointCloud.clear();
		refClusterCentroids.points.clear();

		delete[] plainPointCloud[i];
	}
	//delete[] plainPointCloud;
	//plainPointCloud = nullptr;
	plainPointCloud.clear();
	//delete [] plainCloudSize;
	//delete [] colorPointCloud;
	//colorPointCloud = nullptr;
	colorPointCloud.clear();
	//delete [] clusterBoundingBoxes;
	//clusterBoundingBoxes = nullptr;
	clusterBoundingBoxes.clear();
	//delete [] clusterCentroids;
	//clusterCentroids = nullptr;
	clusterCentroids.clear();
	plainCloudSize.clear();
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
				plainPointCloud[i],
				plainCloudSize[i],
				colorPointCloud[i],
				clusterBoundingBoxes[i],
				clusterCentroids[i]
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
