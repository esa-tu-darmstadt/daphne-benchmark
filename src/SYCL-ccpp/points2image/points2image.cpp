/**
 * Author:  Florian Stock, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2018
 * Author:  Thilo Gabel, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2019
 * License: Apache 2.0 (see attached files)
 */
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdint>
#include <cstring>
#include <string>

#include "points2image.h"
#include "datatypes.h"

points2image::points2image() :
	error_so_far(false),
	max_delta(0.0),
	pointcloud(),
	cameraExtrinsicMat(),
	cameraMat(),
	distCoeff(),
	imageSize(),
	results(),
	computeEnv()
	{}

points2image::~points2image() {}

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
#ifdef EPHOS_TESTDATA_GEN
 	try {
 		datagen_file.open("../../../data/p2i_output.dat.gen", std::ios::binary);
 	} catch (std::ofstream::failure) {
 		std::cerr << "Error opening datagen file" << std::endl;
 		exit(-2);
 	}
#endif
	try {
	// consume the total number of testcases
		testcases = read_number_testcases(input_file);
	} catch (std::ios_base::failure& e) {
		std::cerr << e.what() << std::endl;
		exit(-3);
	}
#ifdef EPHOS_TESTCASE_LIMIT
	testcases = std::min(testcases, (uint32_t)EPHOS_TESTCASE_LIMIT);
#endif
	// prepare the first iteration
	error_so_far = false;
	max_delta = 0.0;

	pointcloud.clear();
	cameraExtrinsicMat.clear();
	cameraMat.clear();
	distCoeff.clear();
	imageSize.clear();
	results.clear();

	// sycl setup
	try {
		computeEnv = ComputeTools::find_compute_platform(EPHOS_PLATFORM_NAME_S, EPHOS_DEVICE_NAME_S,
			EPHOS_DEVICE_TYPE_S);
		std::cout << "SYCL platform: ";
		std::cout << computeEnv.platform.get_info<cl::sycl::info::platform::name>() << std::endl;
		std::cout << "SYCL device: ";
		std::cout << computeEnv.device.get_info<cl::sycl::info::device::name>() << std::endl;
	} catch (std::logic_error& e) {
		std::cerr << "Failed to set up SYCL: " << e.what() << std::endl;
		exit(-3);
	}
	std::cout << "done" << std::endl;
}

void points2image::quit() {
	// close files
	try {
		input_file.close();
	} catch (std::ifstream::failure& e) {
	}
	try {
		output_file.close();

	} catch (std::ofstream::failure& e) {
	}
	try {
		datagen_file.close();
	} catch (std::ofstream::failure& e) {
	}
}

PointsImage points2image::cloud2Image(
	PointCloud& pointcloud,
	Mat44& cameraExtrinsicMat,
	Mat33& cameraMat,
	Vec5& distCoeff,
	ImageSize& imageSize) {

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

	int cloudSize = pointcloud.width*pointcloud.height;
	cl::sycl::buffer<float> cloudBuffer(pointcloud.data, cl::sycl::range<1>(cloudSize*pointcloud.point_step/sizeof(float)));
	cl::sycl::buffer<float> propertyBuffer(cl::sycl::range<1>(cloudSize*2));
	cl::sycl::buffer<int> positionBuffer(cl::sycl::range<1>(cloudSize*2));

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
	try {
		computeEnv.cmdqueue.submit([&](cl::sycl::handler& h) {
			auto cloud = cloudBuffer.get_access<cl::sycl::access::mode::read>(h);
			auto positions = positionBuffer.get_access<cl::sycl::access::mode::write>(h);
			auto properties = propertyBuffer.get_access<cl::sycl::access::mode::write>(h);
			Vec5 coeff = distCoeff;
			int pointStep = pointcloud.point_step/sizeof(float);
			int width = imageSize.width;
			int height = imageSize.height;
			double mCamera[9];
			std::memcpy(mCamera, cameraMat.data, sizeof(double)*9);
			double mProjection[9];
			std::memcpy(mProjection, invR.data, sizeof(double)*9);
			size_t workGroupNo = cloudSize/EPHOS_COMPUTE_GROUP_SIZE + 1;
			h.parallel_for<points2image_kernel>(
				cl::sycl::nd_range<1>(workGroupNo*EPHOS_COMPUTE_GROUP_SIZE, EPHOS_COMPUTE_GROUP_SIZE),
				[=](cl::sycl::nd_item<1> item)
			{
				int iPos = item.get_global_id(0)*2;
				if (iPos < cloudSize*2) {
					int iCloud = item.get_global_id(0)*pointStep;
					Mat13 point0 = {{
						cloud[iCloud + 0],
						cloud[iCloud + 1],
						cloud[iCloud + 2]
					}};
					// apply first transform
					Mat13 point1;
					float intensity = cloud[iCloud + 4];
					for (int row = 0; row < 3; row++) {
						point1.data[row] = invT.data[row];
						for (int col = 0; col < 3; col++) {
							point1.data[row] += point0.data[col]*mProjection[row*3 + col];
						}
					}
					// discard small depth values
					if (point1.data[2] > 2.5) {

						// perform perspective division
						point1.data[0] /= point1.data[2];
						point1.data[1] /= point1.data[2];
						// apply distortion coefficiients
						double radius = point1.data[0]*point1.data[0] + point1.data[1]*point1.data[1];
						double distort = 1 + coeff.data[0]*radius
							+ coeff.data[1]*radius*radius
							+ coeff.data[4]*radius*radius*radius;

						Point2d point2 = {
							point1.data[0]*distort + 2*coeff.data[2]*point1.data[0]*point1.data[1] +
							coeff.data[3]*(radius + 2*point1.data[0]*point1.data[0]),
							point1.data[1]*distort + 2*coeff.data[3]*point1.data[0]*point1.data[1] +
							coeff.data[2]*(radius + 2*point1.data[1]*point1.data[1])
						};
						int point3x = (int)(mCamera[0]*point2.x + mCamera[0*3 + 2] + 0.5);
						int point3y = (int)(mCamera[1*3 + 1]*point2.y + mCamera[1*3 + 2] + 0.5);
						if (0 <= point3x && point3x < width && 0 <= point3y && point3y < height) {
							positions[iPos] = point3x;
							positions[iPos + 1] = point3y;
							properties[iPos] = float(point1.data[2]*100);
							properties[iPos + 1] = intensity;
						} else {
							positions[iPos] = -2;
							positions[iPos + 1] = -2;
						}
					} else {
						positions[iPos] = -1;
						positions[iPos + 1] = -1;
					}
				}
			});
		});
	} catch (cl::sycl::exception& e) {
		std::cerr << e.what();
		exit(-3);
	}
	auto positionStorage = positionBuffer.get_access<cl::sycl::access::mode::read>();
	auto propertyStorage = propertyBuffer.get_access<cl::sycl::access::mode::read>();

	// postprocess results
	for (int iPos = 0; iPos < cloudSize*2; iPos += 2) {
		if (positionStorage[iPos] > -1) {
			int y = positionStorage[iPos + 1];
			int iPixel = positionStorage[iPos] + y*imageSize.width;
			float distance = propertyStorage[iPos];
			if (msg.distance[iPixel] == 0 || distance <= msg.distance[iPixel]) {
				float intensity = propertyStorage[iPos + 1];
				msg.intensity[iPixel] = intensity;
				msg.distance[iPixel] = distance;
				msg.min_height[iPixel] = -1.25;
				msg.max_height[iPixel] = 0;
				if (msg.max_y < y) {
					msg.max_y = y;
				}
				if (msg.min_y > y) {
					msg.min_y = y;
				}
			}
		}
	}
	return msg;
}
