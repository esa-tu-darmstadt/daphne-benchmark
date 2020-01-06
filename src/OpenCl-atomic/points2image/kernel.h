#ifndef EPHOS_KERNEL_H
#define EPHOS_KERNEL_H

#include <cmath>
#include <iostream>
#include <fstream>
#include <string>

#include "benchmark.h"
#include "datatypes.h"

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
};
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



#endif //EPHOS_KERNEL_H
