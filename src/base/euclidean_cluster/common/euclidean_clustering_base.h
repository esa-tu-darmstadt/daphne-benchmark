 /**
 * Author:  Florian Stock, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2018
 * Author:  Thilo Gabel, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2019 - 2020
 * License: Apache 2.0 (see attached files)
 */
#ifndef EPHOS_EUCLIDEAN_CLUSTERING_BASE_H
#define EPHOS_EUCLIDEAN_CLUSTERING_BASE_H


#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>

#include "common/benchmark.h"
#include "datatypes_base.h"

// algorithm parameters
const int _cluster_size_min = 20;
const int _cluster_size_max = 100000;
const bool _pose_estimation = true;

// maximum allowed deviation from the reference data
#define MAX_EPS 0.001

class euclidean_clustering_base : public benchmark {
protected:
	// input point cloud
	std::vector<PlainPointCloud> plainPointCloud;
	// colored point cloud
	std::vector<ColorPointCloud> colorPointCloud;
	// bounding boxes of the input cloud
	std::vector<BoundingboxArray> clusterBoundingBoxes;
	// detected centroids
	std::vector<Centroid> clusterCentroids;
	// the number of testcases that have been read
	int read_testcases = 0;
	// testcase and reference data streams
	std::ifstream input_file, output_file;
	std::ofstream datagen_file;
	// indicates an size related error
	bool error_so_far = false;
	// the measured maximum deviation from the reference data
	double max_delta = 0.0;
public:
	euclidean_clustering_base();
	virtual ~euclidean_clustering_base();
public:
	virtual void init();
	virtual void quit();
	virtual void run(int p = 1);
	virtual bool check_output();
protected:
	/**
	* Helper function for convex hull calculation
	*/
	virtual int sklansky(Point2D** array, int start, int end, int* stack, int nsign, int sign2);
	/**
	* Helper function for calculating the enclosing rectangle with minimum area.
	*/
	virtual void rotatingCalipers( const Point2D* points, int n, float* out );
	/**
	* Helper function that computes the convex hull.
	*/
	virtual void convexHull(
		std::vector<Point2D> _points, std::vector<Point2D>&  _hull, bool clockwise, bool returnPoints);
	/**
	* Computes the rotation angle of the rectangle with minimum area.
	* return: the rotation angle in degrees
	*/
	virtual float minAreaRectAngle(std::vector<Point2D>& points);

	virtual void clusterAndColor(
		const PlainPointCloud& plainPointCloud,
		ColorPointCloud& colorPointCloud,
		int& colorPointCount,
		BoundingboxArray& clusterBoundingBoxes,
		Centroid& clusterCentroids,
		double max_cluster_distance) = 0;
	/**
	* Segments the cloud into categories representing distance ranges from the origin
	* and performs clustering and coloring on the individual categories.
	* plainPointCloud: point cloud
	* colorPointCloud: resulting point cloud
	* clusterBoundingBoxes: resulting bounding boxes
	* in_clusterCentroids: resulting cluster centroids
	* in_max_cluster_distance: distance threshold
	*/
	virtual void segmentByDistance(
		const PlainPointCloud& plainPointCloud,
		ColorPointCloud& colorPointCloud,
		BoundingboxArray& clusterBoundingBoxes,
		Centroid& clusterCentroids);
	/**
	 * Reads the number of testcases in the data set.
	 */
	virtual int read_number_testcases(std::ifstream& input_file);
	/**
	 * Reads the next testcase input data structures.
	 * count: number of testcase datasets to read
	 * return: the number of testcases datasets actually read
	 */
	virtual int read_next_testcases(int count);

	virtual void parsePlainPointCloud(std::ifstream& input_file, PlainPointCloud& cloud);

	virtual void parseColorPointCloud(std::ifstream& input_file, ColorPointCloud& cloud);

	virtual void parseBoundingboxArray(std::ifstream& input_file, BoundingboxArray& bb_array);

	virtual void parseCentroids(std::ifstream& input_file, Centroid& centroids);

	virtual void writeColorPointCloud(std::ofstream& datagen_file, ColorPointCloud& cloud);

	virtual void writeBoundingboxArray(std::ofstream& datagen_file, BoundingboxArray& bb_array);

	virtual void writeCentroids(std::ofstream& datagen_file, Centroid& centroids);
	/**
	 * Reads and compares algorithm outputs with the reference result.
	 * count: the number of outputs to compare
	 */
	virtual void check_next_outputs(int count);
};
#endif // EPHOS_EUCLIDEAN_CLUSTERING_BASE_H
