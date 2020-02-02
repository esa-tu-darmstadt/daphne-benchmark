/**
 * Author:  Florian Stock, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2018
 * Author:  Leonardo Solis, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2018
 * Author:  Thilo Gabel, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2019
 * License: Apache 2.0 (see attachached File)
 */
#ifndef EPHOS_KERNEL_H
#define EPHOS_KERNEL_H

#include <cmath>
#include <iostream>
#include <fstream>
#include <string>

#include "benchmark.h"
#include "datatypes.h"
#include "ocl/host/ocl_header.h"

// maximum allowed deviation from the reference results
#define MAX_EPS 0.001

#define STRINGIZE2(s) #s
#define STRINGIZE(s) STRINGIZE2(s)


// opencl platform hints
#if defined(EPHOS_PLATFORM_HINT)
#define EPHOS_PLATFORM_HINT_S STRINGIZE(EPHOS_PLATFORM_HINT)
#else
#define EPHOS_PLATFORM_HINT_S ""
#endif

#if defined(EPHOS_DEVICE_HINT)
#define EPHOS_DEVICE_HINT_S STRINGIZE(EPHOS_DEVICE_HINT)
#else
#define EPHOS_DEVICE_HINT_S ""
#endif

#if defined(EPHOS_DEVICE_TYPE)
#define EPHOS_DEVICE_TYPE_S STRINGIZE(EPHOS_DEVICE_TYPE)
#else
#define EPHOS_DEVICE_TYPE_S ""
#endif


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
	// opencl members
	OCL_Struct computeEnv;
	cl_program computeProgram;
	cl_kernel transformKernel;
	cl_mem pointcloudBuffer;
	cl_mem counterBuffer;
	cl_mem pixelBuffer;
#ifdef EPHOS_PINNED_MEMORY
	cl_mem pixelHostBuffer;
	PixelData* pixelStorage;
	cl_mem pointcloudHostBuffer;
	float* pointcloudStorage;
#endif

	int maxCloudElementNo = 0;

public:
	points2image();
	~points2image();
public:

	virtual void init();

	virtual void run(int p = 1);

	virtual void quit();

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
private:
	/**
	 * Transforms the given point cloud and produces the result as a two dimensional image.
	 * cloud: input point cloud
	 * cameraExtrinsicMat: perspective projection matrix
	 * distCoeff: distortion coefficients
	 * cameraMat: internal camera matrix
	 * imageSize: output image dimensions
	 * return: output image
	 */
	PointsImage cloud2Image(
		PointCloud2& cloud,
		Mat44& cameraExtrinsicMat,
		Mat33& cameraMat,
		Vec5& distCoeff,
		ImageSize& imageSize);
	/**
	 * Manages the buffers for the transformation of a given point cloud.
	 * cloud: input point cloud
	 */
	void prepare_compute_buffers(PointCloud2& cloud);
	/**
	 * Parses the next point cloud from the input stream.
	 */
	void  parsePointCloud(std::ifstream& input_file, PointCloud2* pointcloud2);
	/**
	 * Parses the next camera extrinsic matrix.
	 */
	void  parseCameraExtrinsicMat(std::ifstream& input_file, Mat44* cameraExtrinsicMat);
	/**
	 * Parses the next camera matrix.
	 */
	void parseCameraMat(std::ifstream& input_file, Mat33* cameraMat);
	/**
	 * Parses the next distance coefficients.
	 */
	void  parseDistCoeff(std::ifstream& input_file, Vec5* distCoeff);
	/**
	 * Parses the next image sizes.
	 */
	void  parseImageSize(std::ifstream& input_file, ImageSize* imageSize);
	/**
	 * Parses the next reference image.
	 */
	void parsePointsImage(std::ifstream& output_file, PointsImage* goldenResult);

	};



#endif //EPHOS_KERNEL_H
