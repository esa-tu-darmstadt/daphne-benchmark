/**
 * Author:  Florian Stock, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2018
 * Author:  Leonardo Solis, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2018
 * Author:  Thilo Gabel, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2019
 * License: Apache 2.0 (see attachached File)
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
#include "common/benchmark.h"



void  points2image::parsePointCloud(std::ifstream& input_file, PointCloud& pointcloud) {
	try {
		input_file.read((char*)&pointcloud.height, sizeof(int32_t));
		input_file.read((char*)&pointcloud.width, sizeof(int32_t));
		input_file.read((char*)&pointcloud.point_step, sizeof(uint32_t));

		prepare_compute_buffers(pointcloud);

		input_file.read((char*)pointcloud.data, pointcloud.height*pointcloud.width*pointcloud.point_step);
    }  catch (std::ifstream::failure) {
		throw std::ios_base::failure("Error reading the next point cloud.");
    }
}

void  points2image::parseCameraExtrinsicMat(std::ifstream& input_file, Mat44& cameraExtrinsicMat) {
	try {
		for (int h = 0; h < 4; h++)
			for (int w = 0; w < 4; w++)
				input_file.read((char*)&cameraExtrinsicMat.data[h][w],sizeof(double));
	} catch (std::ifstream::failure) {
		throw std::ios_base::failure("Error reading the next extrinsic matrix.");
	}
}

void points2image::parseCameraMat(std::ifstream& input_file, Mat33& cameraMat ) {
	try {
	for (int h = 0; h < 3; h++)
		for (int w = 0; w < 3; w++)
			input_file.read((char*)&cameraMat.data[h][w], sizeof(double));
	} catch (std::ifstream::failure) {
		throw std::ios_base::failure("Error reading the next camera matrix.");
    }
}

void  points2image::parseDistCoeff(std::ifstream& input_file, Vec5& distCoeff) {
	try {
		for (int w = 0; w < 5; w++)
			input_file.read((char*)&distCoeff.data[w], sizeof(double));
	} catch (std::ifstream::failure) {
		throw std::ios_base::failure("Error reading the next set of distance coefficients.");
	}
}

void  points2image::parseImageSize(std::ifstream& input_file, ImageSize& imageSize) {
	try {
		input_file.read((char*)&imageSize.width, sizeof(int32_t));
		input_file.read((char*)&imageSize.height, sizeof(int32_t));
	} catch (std::ifstream::failure) {
		throw std::ios_base::failure("Error reading the next image size.");
	}
}

#ifdef EPHOS_TESTCASE_SPARSE
void points2image::parsePointsImage(std::ifstream& output_file, PointsImage& image) {
	try {
		// read data of static size
		int32_t width;
		output_file.read((char*)&width, sizeof(int32_t));
		int32_t height;
		output_file.read((char*)&height, sizeof(int32_t));
		int32_t maxY;
		output_file.read((char*)&maxY, sizeof(int32_t));
		int32_t minY;
		output_file.read((char*)&minY, sizeof(int32_t));
		int pixelNo = width*height;
		image.intensity = new float[pixelNo];
		std::memset(image.intensity, 0, sizeof(float)*pixelNo);
		image.distance = new float[pixelNo];
		std::memset(image.distance, 0, sizeof(float)*pixelNo);
		image.min_height = new float[pixelNo];
		std::memset(image.min_height, 0, sizeof(float)*pixelNo);
		image.max_height = new float[pixelNo];
		std::memset(image.max_height, 0, sizeof(float)*pixelNo);
		image.image_width = width;
		image.image_height = height;
		image.max_y = maxY;
		image.min_y = minY;
		// read sparse image
		int32_t elementNo;
		output_file.read((char*)&elementNo, sizeof(int32_t));
		std::vector<FullPixelData> sparseImage(elementNo);
		output_file.read((char*)sparseImage.data(), sizeof(FullPixelData)*elementNo);
		// create image from sparse image representation
		for (FullPixelData& pixel : sparseImage) {
			int iPixel = pixel.position[1]*width + pixel.position[0];
			image.distance[iPixel] = pixel.depth;
			image.intensity[iPixel] = pixel.intensity;
			image.min_height[iPixel] = pixel.min_height;
			image.max_height[iPixel] = pixel.max_height;
		}
	} catch (std::ios_base::failure &e) {
		throw std::ios_base::failure("Error reading the next reference image.");
	}
}
void points2image::writeSparsePointsImage(std::ofstream& output_file, PointsImage& image) {
	try {
		// read data of static size
		int32_t width = image.image_width;
		output_file.write((char*)&width, sizeof(int32_t));
		int32_t height = image.image_height;
		output_file.write((char*)&height, sizeof(int32_t));
		int32_t maxY = image.max_y;
		output_file.write((char*)&maxY, sizeof(int32_t));
		int32_t minY = image.min_y;
		output_file.write((char*)&minY, sizeof(int32_t));
		// create sparse image representation
		std::vector<FullPixelData> sparseImage;
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				int iPixel = y*width + x;
				if (image.intensity[iPixel] != 0.0f ||
					image.distance[iPixel] != 0.0f ||
					image.min_height[iPixel] != 0.0f ||
					image.max_height[iPixel] != 0.0f) {

					FullPixelData pixel = {
						{ x, y },
						image.distance[iPixel],
						image.intensity[iPixel],
						image.min_height[iPixel],
						image.max_height[iPixel]
					};
					sparseImage.push_back(pixel);
				}
			}
		}
		// write sparse image
		int32_t elementNo = sparseImage.size();
		output_file.write((char*)&elementNo, sizeof(int32_t));
		output_file.write((char*)sparseImage.data(), sizeof(FullPixelData)*elementNo);
	} catch (std::ios_base::failure) {
		throw std::ios_base::failure("Error writing the next reference image.");
	}
}
#else // !EPHOS_TESTCASE_SPARSE
void points2image::parsePointsImage(std::ifstream& output_file, PointsImage& image) {
	try {
		int32_t width;
		output_file.read((char*)&width, sizeof(int32_t));
		int32_t height;
		output_file.read((char*)&height, sizeof(int32_t));
		int32_t maxY;
		output_file.read((char*)&maxY, sizeof(int32_t));
		int32_t minY;
		output_file.read((char*)&minY, sizeof(int32_t));
		int pixelNo = width*height;
		image.intensity = new float[pixelNo];
		image.distance = new float[pixelNo];
		image.min_height = new float[pixelNo];
		image.max_height = new float[pixelNo];
		image.image_width = width;
		image.image_height = height;
		image.max_y = maxY;
		image.min_y = minY;
		// read all pixels
		for (int i = 0; i < pixelNo; i++) {
			float intensity;
			output_file.read((char*)&intensity, sizeof(float));
			float depth;
			output_file.read((char*)&depth, sizeof(float));
			float minHeight;
			output_file.read((char*)&minHeight, sizeof(float));
			float maxHeight;
			output_file.read((char*)&maxHeight, sizeof(float));
			image.distance[i] = depth;
			image.intensity[i] = intensity;
			image.min_height[i] = minHeight;
			image.max_height[i] = maxHeight;
		}
	} catch (std::ios_base::failure) {
		throw std::ios_base::failure("Error reading the next reference image.");
	}
}
#endif // !EPHOS_TESTCASE_SPARSE

int points2image::read_next_testcases(int count)
{
	// and allocate new for the currently required data sizes
	// free counterparts are found in check_next_outputs()
	pointcloud.resize(count);
	cameraExtrinsicMat.resize(count);
	cameraMat.resize(count);
	distCoeff.resize(count);
	imageSize.resize(count);
	results.resize(count);
	/*pointcloud = new PointCloud[count];
	cameraExtrinsicMat = new Mat44[count];
	cameraMat = new Mat33[count];
	distCoeff = new Vec5[count];
	imageSize = new ImageSize[count];
	results = new PointsImage[count];*/

	// iteratively read the data for the test cases
	int i;
	for (i = 0; (i < count) && (read_testcases < testcases); i++,read_testcases++)
	{
		try {
			parsePointCloud(input_file, pointcloud[i]);
			parseCameraExtrinsicMat(input_file, cameraExtrinsicMat[i]);
			parseCameraMat(input_file, cameraMat[i]);
			parseDistCoeff(input_file, distCoeff[i]);
			parseImageSize(input_file, imageSize[i]);
		} catch (std::ios_base::failure& e) {
			std::cerr << e.what() << std::endl;
			exit(-3);
		}
	}

	return i;
}

int points2image::read_number_testcases(std::ifstream& input_file)
{
	// reads the number of testcases in the data stream
	int32_t number;
	try {
		input_file.read((char*)&(number), sizeof(int32_t));
	} catch (std::ifstream::failure) {
		throw std::ios_base::failure("Error reading the number of testcases.");
	}

	return number;
}
void points2image::check_next_outputs(int count)
{
	PointsImage reference;
	// parse the next reference image
	// and compare it to the data generated by the algorithm
	for (int i = 0; i < count; i++)
	{
		std::ostringstream sError;
		int caseErrorNo = 0;
		try {
			parsePointsImage(output_file, reference);
			//writeSparsePointsImage(datagen_file, &reference);
		} catch (std::ios_base::failure& e) {
			std::cerr << e.what() << std::endl;
			exit(-3);
		}
		// detect image size deviation
		if ((results[i].image_height != reference.image_height)
			|| (results[i].image_width != reference.image_width))
		{
			error_so_far = true;
			caseErrorNo += 1;
			sError << " deviating image size: [" << results[i].image_width << " ";
			sError << results[i].image_height << "] should be [";
			sError << reference.image_width << " " << reference.image_height << "]" << std::endl;
		}
		// detect image extend deviation
		if ((results[i].min_y != reference.min_y)
			|| (results[i].max_y != reference.max_y))
		{
			error_so_far = true;
			caseErrorNo += 1;
			sError << " deviating vertical intervall: [" << results[i].min_y << " ";
			sError << results[i].max_y << "] should be [";
			sError << reference.min_y << " " << reference.max_y << "]" << std::endl;
		}
		// compare all pixels
		int pos = 0;
		for (int h = 0; h < reference.image_height; h++)
			for (int w = 0; w < reference.image_width; w++)
			{
				// test for intensity
				float delta = std::fabs(reference.intensity[pos] - results[i].intensity[pos]);
				if (delta > max_delta) {
					max_delta = delta;
				}
				if (delta > MAX_EPS) {
					sError << " at [" << w << " " << h << "]: Intensity " << results[i].intensity[pos];
					sError << " should be " << reference.intensity[pos] << std::endl;
					caseErrorNo += 1;
				}
				// test for distance
				delta = std::fabs(reference.distance[pos] - results[i].distance[pos]);
				if (delta > max_delta) {
					max_delta = delta;
				}
				if (delta > MAX_EPS) {
					sError << " at [" << w << " " << h << "]: Distance " << results[i].distance[pos];
					sError << " should be " << reference.distance[pos] << std::endl;
					caseErrorNo += 1;
				}
				// test for min height
				delta = std::fabs(reference.min_height[pos] - results[i].min_height[pos]);
				if (delta > max_delta) {
					max_delta = delta;
				}
				if (delta > MAX_EPS) {
					sError << " at [" << w << " " << h << "]: Min height " << results[i].min_height[pos];
					sError << " should be " << reference.min_height[pos] << std::endl;
					caseErrorNo += 1;
				}
				// test for max height
				delta = std::fabs(reference.max_height[pos] - results[i].max_height[pos]);
				if (delta > max_delta) {
					max_delta = delta;
				}
				if (delta > MAX_EPS) {
					sError << " at [" << w << " " << h << "]: Max height " << results[i].max_height[pos];
					sError << " should be " << reference.max_height[pos] << std::endl;
					caseErrorNo += 1;
				}
				pos += 1;
			}
		if (caseErrorNo > 0) {
			std::cerr << "Errors for test case " << read_testcases - count + i;
			std::cerr << " (" << caseErrorNo << "):" << std::endl;
			std::cerr << sError.str() << std::endl;
		}
		// free the memory allocated by the reference image read above
		delete[] reference.intensity;
		delete[] reference.distance;
		delete[] reference.min_height;
		delete[] reference.max_height;
		delete[] results[i].intensity;
		delete[] results[i].distance;
		delete[] results[i].min_height;
		delete[] results[i].max_height;
#ifndef EPHOS_PINNED_MEMORY
		delete[] pointcloud[i].data;
#endif
	}
	results.clear();
	cameraExtrinsicMat.clear();
	cameraMat.clear();
	distCoeff.clear();
	imageSize.clear();
	pointcloud.clear();
	/*delete[] results;
	results = nullptr;
	delete[] cameraExtrinsicMat;
	cameraExtrinsicMat = nullptr;
	delete[] cameraMat;
	cameraMat = nullptr;
	delete[] distCoeff;
	distCoeff = nullptr;
	delete[] imageSize;
	imageSize = nullptr;
	delete[] pointcloud;
	pointcloud = nullptr;*/
}

void points2image::run(int p) {
	std::cout << "executing for " << testcases << " test cases" << std::endl;
	// do not measure setup time
	start_timer();
	pause_timer();
	// process all testcases
	while (read_testcases < testcases)
	{
		// read the testcase data, then start the computation
		int count = read_next_testcases(p);
		resume_timer();
		// Set kernel parameters & launch NDRange kernel
		for (int i = 0; i < count; i++)
		{
			results[i] = cloud2Image(pointcloud[i], cameraExtrinsicMat[i], cameraMat[i],
				distCoeff[i], imageSize[i]);
		}
		pause_timer();
		check_next_outputs(count);
	}
	stop_timer();
}

bool points2image::check_output() {
	std::cout << "checking output \n";
	std::cout << "max delta: " << max_delta << "\n";
	if ((max_delta > MAX_EPS) || error_so_far) {
		return false;
	} else {
		return true;
	}
}
// set the external kernel instance used in main()
points2image a;
benchmark& myKernel = a;