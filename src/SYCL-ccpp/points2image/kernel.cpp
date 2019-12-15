/**
 * Author:  Florian Stock, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2018
 * Author:  Thilo Gabel, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2019
 * License: Apache 2.0 (see attachached File)
 */
#include "benchmark.h"
#include "datatypes.h"
#include "sycl/sycl_tools.h"
#include <SYCL/sycl.hpp>
#include <cmath>
#include <iostream>
#include <fstream>
#include <cstring>
#include <ios>

#define MKSTR2(s) #s
#define MKSTR(s) MKSTR2(s)

#ifdef EPHOS_DEVICE_TYPE
#define EPHOS_DEVICE_TYPE_S MKSTR(EPHOS_DEVICE_TYPE)
#else
#define EPHOS_DEVICE_TYPE_S ""
#endif

#ifdef EPHOS_DEVICE_NAME
#define EPHOS_DEVICE_NAME_S MKSTR(EPHOS_DEVICE_NAME)
#else
#define EPHOS_DEVICE_NAME_S ""
#endif

#ifdef EPHOS_PLATFORM_NAME
#define EPHOS_PLATFORM_NAME_S MKSTR(EPHOS_PLATFORM_NAME)
#else
#define EPHOS_PLATFORM_NAME_S ""
#endif

// maximum allowed deviation from the reference results
#define MAX_EPS 0.001
class example_kernel;
class points2image_test;
class points2image_test2;
class points2image_main;
class points2image : public kernel {
private:
	// the number of testcases read
	int read_testcases = 0;
	// testcase and reference data streams
	std::ifstream input_file, output_file;
	// whether critical deviation from the reference data has been detected
	bool error_so_far = false;
	// deviation from the reference data
	double max_delta = 0.0;
	// the point clouds to process in one iteration
	PointCloud2* pointcloud2 = nullptr;
	// the associated camera extrinsic matrices
	Mat44* cameraExtrinsicMat = nullptr;
	// the associated camera intrinsic matrices
	Mat33* cameraMat = nullptr;
	// distance coefficients for the current iteration
	Vec5* distCoeff = nullptr;
	// image sizes for the current iteration
	ImageSize* imageSize = nullptr;
	// Algorithm results for the current iteration
	PointsImage* results = nullptr;
	// sycl state
	cl::sycl::device computeDevice;
	cl::sycl::queue computeQueue;
	size_t computeGroupSize = 0;
public:
	/*
	 * Initializes the kernel. Must be called before run().
	 */
	virtual void init();
	/**
	 * Performs the kernel operations on all input and output data.
	 * p: number of testcases to process in one step
	 */
	virtual void run(int p = 1);
	/**
	 * Finally checks whether all input data has been processed successfully.
	 */
	virtual bool check_output();


	
protected:
	/**
	* Reads the next test cases.
	* count: the number of testcases to read
	* returns: the number of testcases actually read
	*/
	virtual int read_next_testcases(int count);
	/**
	 * Compares the results from the algorithm with the reference data.
	 * count: the number of testcases processed 
	 */
	virtual void check_next_outputs(int count);
	/**
	 * Reads the number of testcases in the data set.
	 */
	int read_number_testcases(std::ifstream& input_file);

	PointsImage pointcloud2_to_image(
		const PointCloud2& pointcloud2,
		const Mat44& cameraExtrinsicMat,
		const Mat33& cameraMat, const Vec5& distCoeff,
		const ImageSize& imageSize);
	
};
/**
 * Parses the next point cloud from the input stream.
 */
void  parsePointCloud(std::ifstream& input_file, PointCloud2* pointcloud2) {
	try {
		input_file.read((char*)&(pointcloud2->height), sizeof(int32_t));
		input_file.read((char*)&(pointcloud2->width), sizeof(int32_t));
		input_file.read((char*)&(pointcloud2->point_step), sizeof(uint32_t));
		pointcloud2->data = new float[pointcloud2->height * pointcloud2->width * pointcloud2->point_step];
		input_file.read((char*)pointcloud2->data, pointcloud2->height * pointcloud2->width * pointcloud2->point_step);
    }  catch (std::ifstream::failure) {
		throw std::ios_base::failure("Error reading the next point cloud.");
    }
}
/**
 * Parses the next camera extrinsic matrix.
 */
void  parseCameraExtrinsicMat(std::ifstream& input_file, Mat44* cameraExtrinsicMat) {
	try {
		for (int h = 0; h < 4; h++)
			for (int w = 0; w < 4; w++)
				input_file.read((char*)&(cameraExtrinsicMat->data[h][w]),sizeof(double));
	} catch (std::ifstream::failure) {
		throw std::ios_base::failure("Error reading the next extrinsic matrix.");		
	}
}
/**
 * Parses the next camera matrix.
 */
void parseCameraMat(std::ifstream& input_file, Mat33* cameraMat ) {
	try {
	for (int h = 0; h < 3; h++)
		for (int w = 0; w < 3; w++)
			input_file.read((char*)&(cameraMat->data[h][w]), sizeof(double));
	} catch (std::ifstream::failure) {
		throw std::ios_base::failure("Error reading the next camera matrix.");
    }
}
/**
 * Parses the next distance coefficients.
 */
void  parseDistCoeff(std::ifstream& input_file, Vec5* distCoeff) {
	try {
		for (int w = 0; w < 5; w++)
			input_file.read((char*)&(distCoeff->data[w]), sizeof(double));
	} catch (std::ifstream::failure) {
		throw std::ios_base::failure("Error reading the next set of distance coefficients.");
	}
}
/**
 * Parses the next image sizes.
 */
void  parseImageSize(std::ifstream& input_file, ImageSize* imageSize) {
	try {
		input_file.read((char*)&(imageSize->width), sizeof(int32_t));
		input_file.read((char*)&(imageSize->height), sizeof(int32_t));
	} catch (std::ifstream::failure) {
		throw std::ios_base::failure("Error reading the next image size.");
	}
}
/**
 * Parses the next reference image.
 */
void parsePointsImage(std::ifstream& output_file, PointsImage* goldenResult) {
	try {
		// read data of static size
		output_file.read((char*)&(goldenResult->image_width), sizeof(int32_t));
		output_file.read((char*)&(goldenResult->image_height), sizeof(int32_t));
		output_file.read((char*)&(goldenResult->max_y), sizeof(int32_t));
		output_file.read((char*)&(goldenResult->min_y), sizeof(int32_t));
		int pos = 0;
		int elements = goldenResult->image_height * goldenResult->image_width;
		goldenResult->intensity = new float[elements];
		goldenResult->distance = new float[elements];
		goldenResult->min_height = new float[elements];
		goldenResult->max_height = new float[elements];
		// read data of variable size
		for (int h = 0; h < goldenResult->image_height; h++)
			for (int w = 0; w < goldenResult->image_width; w++)
			{
				output_file.read((char*)&(goldenResult->intensity[pos]), sizeof(float));
				output_file.read((char*)&(goldenResult->distance[pos]), sizeof(float));
				output_file.read((char*)&(goldenResult->min_height[pos]), sizeof(float));
				output_file.read((char*)&(goldenResult->max_height[pos]), sizeof(float));
				pos++;
			}
	} catch (std::ios_base::failure) {
		throw std::ios_base::failure("Error reading the next reference image.");
	}
}


int points2image::read_next_testcases(int count)
{
	// free the memory that has been allocated in the previous iteration
	// and allocate new for the currently required data sizes
	if (pointcloud2) 
		for (int m = 0; m < count; ++m)
			delete [] pointcloud2[m].data;
	delete [] pointcloud2;
	pointcloud2 = new PointCloud2[count];
	delete [] cameraExtrinsicMat;
	cameraExtrinsicMat = new Mat44[count];
	delete [] cameraMat;
	cameraMat = new Mat33[count];
	delete [] distCoeff;
	distCoeff = new Vec5[count];
	delete [] imageSize;
	imageSize = new ImageSize[count];
	if (results)
	for (int m = 0; m < count; ++m)
	{
		delete [] results[m].intensity;
		delete [] results[m].distance;
		delete [] results[m].min_height;
		delete [] results[m].max_height;
	}
	delete [] results;
	results = new PointsImage[count];
	
	// iteratively read the data for the test cases
	int i;
	for (i = 0; (i < count) && (read_testcases < testcases); i++,read_testcases++)
	{
		try {
			parsePointCloud(input_file, pointcloud2 + i);
			parseCameraExtrinsicMat(input_file, cameraExtrinsicMat + i);
			parseCameraMat(input_file, cameraMat + i);
			parseDistCoeff(input_file, distCoeff + i);
			parseImageSize(input_file, imageSize + i);
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

void points2image::init() {
	std::cout << "init\n";
	std::srand(std::time(nullptr));
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
	computeDevice = SyclTools::selectComputeDevice(EPHOS_PLATFORM_NAME_S, EPHOS_DEVICE_NAME_S, EPHOS_DEVICE_TYPE_S);
	std::string deviceName = computeDevice.get_info<cl::sycl::info::device::name>();
	std::cout << "Compute device name: " << deviceName << std::endl;
	computeGroupSize = computeDevice.get_info<cl::sycl::info::device::max_work_group_size>();
	computeQueue = cl::sycl::queue(computeDevice);
	
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
PointsImage points2image::pointcloud2_to_image(
	const PointCloud2& pointcloud2,
	const Mat44& cameraExtrinsicMat,
	const Mat33& cameraMat, const Vec5& distCoeff,
	const ImageSize& imageSize)
{
	int cloudSize = pointcloud2.width*pointcloud2.height;
	cl::sycl::buffer<float> cloudBuffer(pointcloud2.data, cl::sycl::range<1>(cloudSize*pointcloud2.point_step/sizeof(float)));
	cl::sycl::buffer<float> propertyBuffer(cl::sycl::range<1>(cloudSize*2));
	cl::sycl::buffer<int> positionBuffer(cl::sycl::range<1>(cloudSize*2));
	
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
	// initialize buffers
	// apply the algorithm
	// perform transformations
	try {
	computeQueue.submit([&](cl::sycl::handler& h) {
		auto cloud = cloudBuffer.get_access<cl::sycl::access::mode::read>(h);
		auto positions = positionBuffer.get_access<cl::sycl::access::mode::write>(h);
		auto properties = propertyBuffer.get_access<cl::sycl::access::mode::write>(h);
		Vec5 coeff = distCoeff;
		int pointStep = pointcloud2.point_step/sizeof(float);
		int width = imageSize.width;
		int height = imageSize.height;
		double mCamera[9];
		std::memcpy(mCamera, cameraMat.data, sizeof(double)*9);
		double mProjection[9];
		std::memcpy(mProjection, invR.data, sizeof(double)*9);
		size_t workGroupNo = cloudSize/computeGroupSize + 1;
		h.parallel_for<points2image_main>(
			cl::sycl::nd_range<1>(workGroupNo*computeGroupSize, computeGroupSize),
			[=](cl::sycl::nd_item<1> item) {
		//h.parallel_for<points2image_main>(cl::sycl::range<1>(cloudSize), [=](cl::sycl::id<1> item) {
			//int iPos = item.get(0)*2;

			//int iCloud = item.get(0)*pointStep;

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
				/*double tmpx = point.data[0]/point.data[2];
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
				int py = int(imagepoint.y + 0.5);*/
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
						point1.data[0]*distort + 2*coeff.data[2]*point1.data[0]*point1.data[1] + coeff.data[3]*(radius + 2*point1.data[0]*point1.data[0]),
						point1.data[1]*distort + 2*coeff.data[3]*point1.data[0]*point1.data[1] + coeff.data[2]*(radius + 2*point1.data[1]*point1.data[1])
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
	computeQueue.wait();
	} catch (cl::sycl::exception& e) {
		std::cerr << e.what() << std::endl;
		exit(-3);
	}
	auto positionStorage = positionBuffer.get_access<cl::sycl::access::mode::read>();
	auto propertyStorage = propertyBuffer.get_access<cl::sycl::access::mode::read>();
	/*{
		for (int i = 0; i < 32; i++) {
			if (positionStorage[i] == -1) {
				std::cout << "Wrote 2 to position buffer" << std::endl;
			}
		}
	}*/
	// postprocess results
	for (int iPos = 0; iPos < cloudSize*2; iPos += 2) {
		if (positionStorage[iPos] > -1) {
			int y = positionStorage[iPos + 1];
			int iPixel = positionStorage[iPos] + y*imageSize.width;
			float distance = propertyStorage[iPos];
			if (msg.distance[iPixel] == 0 || distance <= msg.distance[iPixel]) {
				float intensity = propertyStorage[iPos + 1];

				//properties[iPos] == msg.distance[iPixel] && msg.intensity[iPixel] < properties[iPos + 1]) {

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



	//apply the algorithm for each point in the cloud
	/*for (uint32_t y = 0; y < pointcloud2.height; ++y) {
		for (uint32_t x = 0; x < pointcloud2.width; ++x) {
			// the start of the current point in the cloud to process
			int iPoint = (x + y*pointcloud2.width);
			int iPos = iPoint*2;
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
				if (positionStorage[iPos] != -1) {
					std::cout << "Position " << iPos << " ("<< positionStorage[iPos] << ") is not discarded" << std::endl;
				} else {
					//std::cout << "Position " << iPos << " is correctly discarded" << std::endl;
				}
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
			imagepoint.x = tmpx;
			imagepoint.y = tmpy;
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
				if (positionStorage[iPos] != px || positionStorage[iPos + 1] != py) {
					std::cout << "Position " << iPos << " (" << positionStorage[iPos] << " " << positionStorage[iPos + 1] << ")";
					std::cout << " is not " << px << " " << py << std::endl;
				}
			}
		}
	}*/
	return msg;
}


void points2image::run(int p) {
	// pause while reading and comparing data
	// only run the timer when the algorithm is active
	pause_func();
	int count = read_next_testcases(p);
	for (int i = 0; i < count; i++)
	{
		results[i] = pointcloud2_to_image(
			pointcloud2[i],
			cameraExtrinsicMat[i],
			cameraMat[i], distCoeff[i],
			imageSize[i]);
	}
	if (results) {
		for (int m = 0; m < count; ++m)
		{
			delete [] results[m].intensity;
			delete [] results[m].distance;
			delete [] results[m].min_height;
			delete [] results[m].max_height;
		}
		delete [] results;
	}
	results = new PointsImage[count];

	while (true)
	{
		unpause_func();
		// run the algorithm for each input data set
		for (int i = 0; i < count; i++)
		{
			results[i] = pointcloud2_to_image(
				pointcloud2[i],
				cameraExtrinsicMat[i],
				cameraMat[i], distCoeff[i],
				imageSize[i]);
		}
		pause_func();
		// compare with the reference data
		check_next_outputs(count);
		if (read_testcases < testcases) {
			count = read_next_testcases(p);
		} else {
			break;
		}
	}

// 	pause_func();
// 	while (read_testcases < testcases)
// 	{
// 		int count = read_next_testcases(p);
// 		unpause_func();
// 		// run the algorithm for each input data set
// 		for (int i = 0; i < count; i++)
// 		{
// 			results[i] = pointcloud2_to_image(pointcloud2[i],
// 								cameraExtrinsicMat[i],
// 								cameraMat[i], distCoeff[i],
// 								imageSize[i]);
// 		}
// 		pause_func();
// 		// compare with the reference data
// 		check_next_outputs(count);
// 	}
}

void points2image::check_next_outputs(int count)
{
	PointsImage reference;
	// parse the next reference image
	// and compare it to the data generated by the algorithm
	for (int i = 0; i < count; i++)
	{
		try {
			parsePointsImage(output_file, &reference);
		} catch (std::ios_base::failure& e) {
			std::cerr << e.what() << std::endl;
			exit(-3);
		}
		// detect image size deviation
		if ((results[i].image_height != reference.image_height)
			|| (results[i].image_width != reference.image_width))
		{
			error_so_far = true;
		}
		// detect image extend deviation
		if ((results[i].min_y != reference.min_y)
			|| (results[i].max_y != reference.max_y))
		{
			error_so_far = true;
		}
		// compare all pixels
		int pos = 0;
		for (int h = 0; h < reference.image_height; h++)
			for (int w = 0; w < reference.image_width; w++)
			{
				// compare members individually and detect deviations
				if (std::fabs(reference.intensity[pos] - results[i].intensity[pos]) > max_delta)
					max_delta = fabs(reference.intensity[pos] - results[i].intensity[pos]);
				if (std::fabs(reference.distance[pos] - results[i].distance[pos]) > max_delta)
					max_delta = fabs(reference.distance[pos] - results[i].distance[pos]);
				if (std::fabs(reference.min_height[pos] - results[i].min_height[pos]) > max_delta)
					max_delta = fabs(reference.min_height[pos] - results[i].min_height[pos]);
				if (std::fabs(reference.max_height[pos] - results[i].max_height[pos]) > max_delta)
					max_delta = fabs(reference.max_height[pos] - results[i].max_height[pos]);
				pos++;
			}
		// free the memory allocated by the reference image read above
		delete [] reference.intensity;
		delete [] reference.distance;
		delete [] reference.min_height;
		delete [] reference.max_height;
	}
}

bool points2image::check_output() {
	std::cout << "checking output \n";
	// complement to init()
	input_file.close();
	output_file.close();
	std::cout << "max delta: " << max_delta << "\n";
	if ((max_delta > MAX_EPS) || error_so_far) {
		return false;
	} else {
		return true;
	}
}
// set the external kernel instance used in main()
points2image a = points2image();
kernel& myKernel = a;
