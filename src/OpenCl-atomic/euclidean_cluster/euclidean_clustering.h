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
#include "common/compute_tools.h"

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

#define NUMWORKITEMS_PER_WORKGROUP_STRING STRINGIZE(NUMWORKITEMS_PER_WORKGROUP)

// algorithm parameters
const int _cluster_size_min = 20;
const int _cluster_size_max = 100000;
const bool _pose_estimation = true;

// maximum allowed deviation from the reference data
#define MAX_EPS 0.001

class euclidean_clustering : public benchmark {
private:
	// input point cloud
	PointCloud *plainPointCloud = nullptr;
	// colored point cloud
	ColorPointCloud *colorPointCloud = nullptr;
	// bounding boxes of the input cloud
	BoundingboxArray* clusterBoundingBoxes = nullptr;
	// detected centroids
	Centroid *clusterCentroids = nullptr;
	// number of elements in the point clouds
	std::vector<int> plainCloudSize;
	// the number of testcases that have been read
	int read_testcases = 0;
	// testcase and reference data streams
	std::ifstream input_file, output_file;
	// indicates an size related error
	bool error_so_far = false;
	// the measured maximum deviation from the reference data
	double max_delta = 0.0;
	// compute resources
	ComputeEnv computeEnv;
	cl::Kernel distanceMatrixKernel;
	cl::Kernel buildClusterKernel;

public:
	virtual void init();
	virtual void quit();
	virtual void run(int p = 1);
	virtual bool check_output();
protected:
	/**
	* Helper function for convex hull calculation
	*/
	int sklansky(Point2D** array, int start, int end, int* stack, int nsign, int sign2);
	/**
	* Helper function for calculating the enclosing rectangle with minimum area.
	*/
	void rotatingCalipers( const Point2D* points, int n, float* out );
	/**
	* Helper function that computes the convex hull.
	*/
	void convexHull(
		std::vector<Point2D> _points, std::vector<Point2D>&  _hull, bool clockwise, bool returnPoints);
	/**
	* Computes the rotation angle of the rectangle with minimum area.
	* return: the rotation angle in degrees
	*/
	float minAreaRectAngle(std::vector<Point2D>& points);
	/**
	* Finds all clusters in the given point cloud that are conformant to the given parameters.
	* cloud: point cloud to cluster
	* cloudSize: number of cloud elements
	* tolerance: search radius around a single point
	* clusters: list of resulting clusters
	* min_pts_per_cluster: lower cluster size restriction
	* max_pts_per_cluster: higher cluster size restriction
	*/
	void extractEuclideanClusters (
		const PointCloud cloud,
		int cloudSize,
		float tolerance,
		std::vector<PointIndices> &clusters,
		unsigned int min_pts_per_cluster,
		unsigned int max_pts_per_cluster);

	void extract (
		const PointCloud input_,
		int cloudSize,
		std::vector<PointIndices> &clusters,
		#if defined (DOUBLE_FP)
		double cluster_tolerance_
		#else
		float cluster_tolerance_
		#endif
		);


	void clusterAndColor(
		const PointCloud plainPointCloud,
		int plainCloudSize,
		ColorPointCloud *colorPointCloud,
		BoundingboxArray *in_clusterBoundingBoxes,
		Centroid *in_clusterCentroids,
		#if defined (DOUBLE_FP)
		double in_max_cluster_distance
		#else
		float in_max_cluster_distance
		#endif
	);
	/**
	* Segments the cloud into categories representing distance ranges from the origin
	* and performs clustering and coloring on the individual categories.
	* plainPointCloud: point cloud
	* cloudSize: number of points in cloud
	* colorPointCloud: resulting point cloud
	* clusterBoundingBoxes: resulting bounding boxes
	* in_clusterCentroids: resulting cluster centroids
	* in_max_cluster_distance: distance threshold
	*/
	void segmentByDistance(
		const PointCloud plainPointCloud,
		int plainCloudSize,
		ColorPointCloud *colorPointCloud,
		BoundingboxArray *in_clusterBoundingBoxes,
		Centroid *in_clusterCentroids);
	/**
	 * Reads the number of testcases in the data set.
	 */
	int read_number_testcases(std::ifstream& input_file);
	/**
	 * Reads the next testcase input data structures.
	 * count: number of testcase datasets to read
	 * return: the number of testcases datasets actually read
	 */
	int read_next_testcases(int count);

	void parsePointCloud(std::ifstream& input_file, PointCloud *cloud, int *cloudSize);

	void parseOutCloud(std::ifstream& input_file, ColorPointCloud *cloud);

	void parseBoundingboxArray(std::ifstream& input_file, BoundingboxArray *bb_array);

	void parseCentroids(std::ifstream& input_file, Centroid *centroids);


	/**
	 * Reads and compares algorithm outputs with the reference result.
	 * count: the number of outputs to compare
	 */
	void check_next_outputs(int count);
};

/**
 * Extension for for point comparison
 */
// struct CHullCmpPoints
// {
// 	/**
// 	 * Compares two points.
// 	 * Performs a primary test for the x component
// 	 * and a secondary test for the y component.
// 	 */
// 	bool operator()(const Point2D* p1, const Point2D* p2) const
// 	{
// 		return p1->x < p2->x || (p1->x == p2->x && p1->y < p2->y);
// 	}
// };