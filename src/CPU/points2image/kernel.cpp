/**
 * Author:  Florian Stock, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2018
 * License: Apache 2.0 (see attachached File)
 */
#include "benchmark.h"
#include "datatypes.h"
#include "kernel.h"
#include <cmath>
#include <iostream>
#include <fstream>
#include <cstring>
#include <ios>

void points2image::init() {
	std::cout << "init\n";
	
	// open testcase and reference data streams
	input_file.exceptions ( std::ifstream::failbit | std::ifstream::badbit );
	output_file.exceptions ( std::ifstream::failbit | std::ifstream::badbit );
	try {
		input_file.open("../../../data/p2i_input.dat", std::ios::binary);
	} catch (std::ifstream::failure) {
		std::cerr << "Error opening the input data file" << std::endl;
		exit(-2);
	}
	try {
		output_file.open("../../../data/p2i_output.dat", std::ios::binary);
	} catch (std::ifstream::failure) {
		std::cerr << "Error opening the output data file" << std::endl;
		exit(-2);
	}
	try {
	// consume the total number of testcases
		testcases = read_number_testcases(input_file);
	} catch (std::ios_base::failure& e) {
		std::cerr << e.what() << std::endl;
		exit(-3);
	}
	
	// prepare the first iteration
	error_so_far = false;
	max_delta = 0.0;
	pointcloud2 = nullptr;
	cameraExtrinsicMat = nullptr;
	cameraMat = nullptr;
	distCoeff = nullptr;
	imageSize = nullptr;
	results = nullptr;

	std::cout << "done\n" << std::endl;
}
/**
 * This code is extracted from Autoware, file:
 * ~/Autoware/ros/src/sensing/fusion/packages/points2image/lib/points_image/points_image.cpp
 * It uses the test data that has been read before and applies the linked algorithm.
 * pointcloud2: cloud of points to transform
 * cameraExtrinsicMat: camera matrix used for transformation
 * cameraMat: camera matrix used for transformation
 * distCoeff: distance coefficients for cloud transformation
 * imageSize: the size of the resulting image
 * returns: the two dimensional image of transformed points
 */
PointsImage pointcloud2_to_image(
	const PointCloud2& pointcloud2,
	const Mat44& cameraExtrinsicMat,
	const Mat33& cameraMat, const Vec5& distCoeff,
	const ImageSize& imageSize)
{
	// initialize the resulting image data structure
	int w = imageSize.width;
	int h = imageSize.height;
	PointsImage msg;
	msg.intensity = new float[w*h];
	std::memset(msg.intensity, 0, sizeof(float)*w*h);
	msg.distance = new float[w*h];
	std::memset(msg.distance, 0, sizeof(float)*w*h);
	msg.min_height = new float[w*h];
	std::memset(msg.min_height, 0, sizeof(float)*w*h);
	msg.max_height = new float[w*h];
	std::memset(msg.max_height, 0, sizeof(float)*w*h);
	msg.max_y = -1;
	msg.min_y = h;
	msg.image_height = imageSize.height;
	msg.image_width = imageSize.width;
	
	// prepare cloud data pointer to read the data correctly
	uintptr_t cp = (uintptr_t)pointcloud2.data;
	
	// preprocess the given matrices
	// transposed 3x3 camera extrinsic matrix
	Mat33 invR;
	for (int row = 0; row < 3; row++)
		for (int col = 0; col < 3; col++)
			invR.data[row][col] = cameraExtrinsicMat.data[col][row];
	// translation vector: (transposed camera extrinsic matrix)*(fourth column of camera extrinsic matrix)
	Mat13 invT;
	for (int row = 0; row < 3; row++) {
		invT.data[row] = 0.0;
		for (int col = 0; col < 3; col++)
			invT.data[row] -= invR.data[row][col] * cameraExtrinsicMat.data[col][3];
	}
	// apply the algorithm for each point in the cloud
	for (uint32_t y = 0; y < pointcloud2.height; ++y) {
		for (uint32_t x = 0; x < pointcloud2.width; ++x) {
			// the start of the current point in the cloud to process
			float* fp = (float *)(cp + (x + y*pointcloud2.width) * pointcloud2.point_step);
			double intensity = fp[4];

			Mat13 point, point2;
			point2.data[0] = double(fp[0]);
			point2.data[1] = double(fp[1]);
			point2.data[2] = double(fp[2]);
			
			// start the the predetermined translation
			for (int row = 0; row < 3; row++) {
				point.data[row] = invT.data[row];
			// add the transformed cloud point
			for (int col = 0; col < 3; col++) 
				point.data[row] += point2.data[col] * invR.data[row][col];
			}
			// discard points with small depth values
			if (point.data[2] <= 2.5) {
				continue;
			}
			// perform perspective division
			double tmpx = point.data[0]/point.data[2];
			double tmpy = point.data[1]/point.data[2];
			// apply the distance coefficients
			double r2 = tmpx * tmpx + tmpy * tmpy;
			double tmpdist = 1 + distCoeff.data[0] * r2
			+ distCoeff.data[1] * r2 * r2
			+ distCoeff.data[4] * r2 * r2 * r2;

			Point2d imagepoint;
			imagepoint.x = tmpx * tmpdist
			+ 2 * distCoeff.data[2] * tmpx * tmpy
			+ distCoeff.data[3] * (r2 + 2 * tmpx * tmpx);
			imagepoint.y = tmpy * tmpdist
			+ distCoeff.data[2] * (r2 + 2 * tmpy * tmpy)
			+ 2 * distCoeff.data[3] * tmpx * tmpy;
			
			// apply the camera matrix (camera intrinsics) and end up with a two dimensional point
			imagepoint.x = cameraMat.data[0][0] * imagepoint.x + cameraMat.data[0][2];
			imagepoint.y = cameraMat.data[1][1] * imagepoint.y + cameraMat.data[1][2];
			int px = int(imagepoint.x + 0.5);
			int py = int(imagepoint.y + 0.5);
			
			// continue with points that landed inside image bounds
			if(0 <= px && px < w && 0 <= py && py < h)
			{
				// target pixel index linearization
				int pid = py * w + px;
				// replace unset pixels as well as pixels with a higher distance value
				if(msg.distance[pid] == 0.0 ||
					msg.distance[pid] >= float(point.data[2] * 100.0))
				{
					// make the result always deterministic and independent from the point order
					// in case two points get the same distance, take the one with higher intensity
					if (((msg.distance[pid] == float(point.data[2] * 100.0)) &&  msg.intensity[pid] < float(intensity)) ||
						(msg.distance[pid] > float(point.data[2] * 100.0)) ||
						msg.distance[pid] == 0) 
					{
						msg.intensity[pid] = float(intensity);
					}
					msg.distance[pid] = float(point.data[2] * 100.0);
					msg.min_height[pid] = -1.25;
					msg.max_height[pid] = 0;
					// update image usage extends
					msg.max_y = py > msg.max_y ? py : msg.max_y;
					msg.min_y = py < msg.min_y ? py : msg.min_y;
				}
			}
		}
	}
	return msg;
}


void points2image::run(int p) {
	// pause while reading and comparing data
	// only run the timer when the algorithm is active
	pause_func();
	while (read_testcases < testcases)
	{
		int count = read_next_testcases(p);
		unpause_func();
		// run the algorithm for each input data set
		for (int i = 0; i < count; i++)
		{
			results[i] = pointcloud2_to_image(pointcloud2[i],
								cameraExtrinsicMat[i],
								cameraMat[i], distCoeff[i],
								imageSize[i]);
		}
		pause_func();
		// compare with the reference data
		check_next_outputs(count);
	}
}
