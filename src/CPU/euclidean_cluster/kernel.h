/**
 * Author:  Florian Stock, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2018
 * License: Apache 2.0 (see attachached File)
 */
#ifndef EPHOS_KERNEL_H
#define EPHOS_KERNEL_H

#include <iostream>
#include <fstream>
#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>

#include "benchmark.h"
#include "datatypes.h"

// algorithm parameters
const int _cluster_size_min = 20;
const int _cluster_size_max = 100000;
const bool _pose_estimation = true;

// maximum allowed deviation from the reference data
#define MAX_EPS 0.001

class euclidean_clustering : public kernel {
private:
	// input point cloud
	PointCloud *in_cloud_ptr = nullptr;
	// colored point cloud
	PointCloudRGB *out_cloud_ptr = nullptr;
	// bounding boxes of the input cloud
	BoundingboxArray *out_boundingbox_array = nullptr;
	// detected centroids
	Centroid *out_centroids = nullptr;
	// the number of testcases that have been read
	int read_testcases = 0;
	// testcase and reference data streams
	std::ifstream input_file, output_file;
	// indicates an size related error
	bool error_so_far = false;
	// the measured maximum deviation from the reference data
	double max_delta = 0.0;
public:
	virtual void init();
	virtual void run(int p = 1);
	virtual bool check_output();
protected:

	/**
	 * Finds all clusters in the given point cloud that are conformant to the given parameters.
	 * cloud: point cloud to cluster
	 * tolerance: search radius around a single point
	 * clusters: list of resulting clusters
	 * min_pts_per_cluster: lower cluster size restriction
	 * max_pts_per_cluster: higher cluster size restriction
	 */
	void extractEuclideanClusters (const PointCloud &cloud,
		float tolerance, std::vector<PointIndices> &clusters,
		unsigned int min_pts_per_cluster, unsigned int max_pts_per_cluster);

	/**
	 * Performs clustering and coloring on a point cloud
	 */
	void clusterAndColor(const PointCloud *in_cloud_ptr,
		PointCloudRGB *out_cloud_ptr,
		BoundingboxArray *in_out_boundingbox_array,
		Centroid *in_out_centroids,
		double in_max_cluster_distance);


	/**
	 * Computes euclidean clustering and sorts the resulting clusters.
	 */
	void extract (const PointCloud *input_, std::vector<PointIndices> &clusters,
		double cluster_tolerance_);
	/**
	 * Segments the cloud into categories representing distance ranges from the origin
	 * and performs clustering and coloring on the individual categories.
	 * in_cloud_ptr: point cloud
	 * out_cloud_ptr: resulting point cloud
	 * out_boundingbox_array: resulting bounding boxes
	 * in_out_centroids: resulting cluster centroids
	 * in_max_cluster_distance: distance threshold
	 */
	void segmentByDistance(const PointCloud *in_cloud_ptr,
		PointCloudRGB *out_cloud_ptr,
		BoundingboxArray *in_out_boundingbox_array,
		Centroid *in_out_centroids,
		double in_max_cluster_distance);
	/**
	 * Reads the number of testcases in the data set.
	 */
	int read_number_testcases(std::ifstream& input_file);
	/**
	 * Reads the next testcase input data structures.
	 * count: number of testcase datasets to read
	 * return: the number of testcases datasets actually read
	 */
	virtual int read_next_testcases(int count);
	/**
	 * Reads and compares algorithm outputs with the reference result.
	 * count: the number of outputs to compare
	 */
	virtual void check_next_outputs(int count);
};

/**
 * Reads the next point cloud.
 */
void parsePointCloud(std::ifstream& input_file, PointCloud *cloud);
/**
 * Reads the next reference cloud result.
 */
void parseOutCloud(std::ifstream& input_file, PointCloudRGB *cloud);
/**
 * Reads the next reference bounding boxes.
 */
void parseBoundingboxArray(std::ifstream& input_file, BoundingboxArray *bb_array);
/*
 * Reads the next reference centroids.
 */
void parseCentroids(std::ifstream& input_file, Centroid *centroids);


/**
 * Helper function for point comparison
 */
bool compareRGBPoints (const PointRGB &a, const PointRGB &b);
/**
 * Helper function for point comparison
 */
bool comparePoints (const PointDouble &a, const PointDouble &b);
/**
 * Helper function for bounding box comparison
 */
bool compareBBs (const Boundingbox &a, const Boundingbox &b);
/**
 * Helper function that compares cluster sizes.
 */
bool comparePointClusters (const PointIndices &a, const PointIndices &b);
/**
 * Helper function for convex hull comparison.
 */
bool compareConvexHullPoints(const Point2D* p1, const Point2D* p2);
#endif
