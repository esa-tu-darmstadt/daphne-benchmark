/**
 * Author:  Florian Stock, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2018
 * Author:  Thilo Gabel, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2019 - 2020
 * License: Apache 2.0 (see attached files)
 */
#ifndef EPHOS_POINTS2IMAGE_BASE_H
#define EPHOS_POINTS2IMAGE_BASE_H

#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>

#include "datatypes_base.h"
#include "common/benchmark.h"

// maximum allowed deviation from the reference results
#define MAX_EPS 0.001

class points2image_base : public benchmark {
protected:
	// the number of testcases read
	int read_testcases = 0;
	// testcase and reference data streams
	std::ifstream input_file, output_file;
	std::ofstream datagen_file;
	// whether critical deviation from the reference data has been detected
	bool error_so_far = false;
	// deviation from the reference data
	double max_delta = 0.0;
	// the point clouds to process in one iteration
	std::vector<PointCloud> pointcloud;
	// the associated camera extrinsic matrices
	std::vector<Mat44> cameraExtrinsicMat;
	// the associated camera intrinsic matrices
	std::vector<Mat33> cameraMat;
	// distance coefficients for the current iteration
	std::vector<Vec5> distCoeff;
	// image sizes for the current iteration
	std::vector<ImageSize> imageSize;
	// Algorithm results for the current iteration
	std::vector<PointsImage> results;

protected:
	points2image_base();
	virtual ~points2image_base();

public:
	virtual void init();
	virtual void quit();
	virtual void run(int p = 1);
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
	virtual int read_number_testcases(std::ifstream& input_file);
	/**
	 * Transforms the given point cloud and produces the result as a two dimensional image.
	 * cloud: input point cloud
	 * cameraExtrinsicMat: perspective projection matrix
	 * distCoeff: distortion coefficients
	 * cameraMat: internal camera matrix
	 * imageSize: output image dimensions
	 * return: output image
	 */
	virtual PointsImage cloud2Image(
		PointCloud& cloud,
		Mat44& cameraExtrinsicMat,
		Mat33& cameraMat,
		Vec5& distCoeff,
		ImageSize& imageSize) = 0;
	/**
	 * Parses the next point cloud from the input stream.
	 */
	virtual void parsePointCloud(std::ifstream& input_file, PointCloud& pointcloud);
	/**
	 * Parses the next camera extrinsic matrix.
	 */
	virtual void parseCameraExtrinsicMat(std::ifstream& input_file, Mat44& cameraExtrinsicMat);
	/**
	 * Parses the next camera matrix.
	 */
	virtual void parseCameraMat(std::ifstream& input_file, Mat33& cameraMat);
	/**
	 * Parses the next distance coefficients.
	 */
	virtual void parseDistCoeff(std::ifstream& input_file, Vec5& distCoeff);
	/**
	 * Parses the next image sizes.
	 */
	virtual void parseImageSize(std::ifstream& input_file, ImageSize& imageSize);
	/**
	 * Parses the next reference image.
	 */
	virtual void parsePointsImage(std::ifstream& output_file, PointsImage& image);
	/**
	 * Outputs a sparse image representation to the given stream.
	 */
	virtual void writePointsImage(std::ofstream& output_file, PointsImage& image);

};

#endif // EPHOS_POINTS2IMAGE_BASE_H
