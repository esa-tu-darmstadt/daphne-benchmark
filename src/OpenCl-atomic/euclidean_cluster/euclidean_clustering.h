 /**
 * Author:  Florian Stock, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2018
 * Author:  Leonardo Solis, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2018
 * Author:  Thilo Gabel, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2019 - 2020
 * License: Apache 2.0 (see attachached File)
 */
#ifndef EPHOS_EUCLIDEAN_CLUSTERING_H
#define EPHOS_EUCLIDEAN_CLUSTERING_H


#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>

#include "common/benchmark.h"
#include "datatypes.h"
#include "common/compute_tools.h"
#include "common/euclidean_clustering_base.h"

#define STRINGIFY2(s) #s
#define STRINGIFY(s) STRINGIFY2(s)

// opencl platform hints
#if defined(EPHOS_PLATFORM_HINT)
#define EPHOS_PLATFORM_HINT_S STRINGIFY(EPHOS_PLATFORM_HINT)
#else
#define EPHOS_PLATFORM_HINT_S ""
#endif

#if defined(EPHOS_DEVICE_HINT)
#define EPHOS_DEVICE_HINT_S STRINGIFY(EPHOS_DEVICE_HINT)
#else
#define EPHOS_DEVICE_HINT_S ""
#endif

#if defined(EPHOS_DEVICE_TYPE)
#define EPHOS_DEVICE_TYPE_S STRINGIFY(EPHOS_DEVICE_TYPE)
#else
#define EPHOS_DEVICE_TYPE_S ""
#endif

// maximum allowed deviation from the reference data
#define MAX_EPS 0.001

extern const char* radius_search_ocl_kernel_soure;

class euclidean_clustering : public euclidean_clustering_base {
private:
	// compute resources
	ComputeEnv computeEnv;
	cl::Kernel distanceMatrixKernel;
	cl::Kernel radiusSearchKernel;
	cl::Buffer seedQueueBuffer;
	cl::Buffer processedBuffer;
	cl::Buffer distanceBuffer;
	cl::Buffer pointCloudBuffer;
	cl::Buffer seedQueueLengthBuffer;
	int maxSeedQueueLength = -1;
public:
	euclidean_clustering();
	virtual ~euclidean_clustering();
public:
	virtual void init();
	virtual void quit();
protected:
	virtual void clusterAndColor(
		const PlainPointCloud& plainPointCloud,
		ColorPointCloud& colorPointCloud,
		BoundingboxArray& clusterBoundingBoxes,
		Centroid& clusterCentroids,
		double max_cluster_distance) override;

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
		Centroid& clusterCentroids) override;
private:
	/**
	* Finds all clusters in the given point cloud that are conformant to the given parameters.
	* cloud: point cloud to cluster
	* cloudSize: number of cloud elements
	* tolerance: search radius around a single point
	* clusters: list of resulting clusters
	* min_pts_per_cluster: lower cluster size restriction
	* max_pts_per_cluster: higher cluster size restriction
	*/
	void extractEuclideanClusters(
		const PlainPointCloud& plainPointCloud,
		float tolerance,
		std::vector<PointIndices> &clusters,
		unsigned int min_pts_per_cluster,
		unsigned int max_pts_per_cluster);

	void extract(
		const PlainPointCloud& plainPointCloud,
		std::vector<PointIndices> &clusters,
		double tolerance);
	/**
	 * Manages compute resources.
	 * plainPointCloud: biggest point cloud
	 * cloudSize: number of cloud elements
	 */
	void prepare_compute_buffers(const PlainPointCloud& plainPointCloud, int cloudSize);
};
#endif // EPHOS_EUCLIDEAN_CLUSTERING_H
