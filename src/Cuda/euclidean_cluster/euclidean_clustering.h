 /**
 * Author:  Florian Stock, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2018
 * Author:  Thilo Gabel, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2019 - 2020
 * License: Apache 2.0 (see attached files)
 */
#ifndef EPHOS_EUCLIDEAN_CLUSTERING_H
#define EPHOS_EUCLIDEAN_CLUSTERING_H


#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>
#include <cstring>

#include "common/euclidean_clustering_base.h"

class euclidean_clustering : public euclidean_clustering_base {

public:
	euclidean_clustering() : euclidean_clustering_base() {}
	virtual ~euclidean_clustering() {}
public:
	virtual void init() override;
	virtual void quit() override;
protected:

	//void initRadiusSearch(const PlainPointCloud& points, bool**  sqr_distances, float radius);

	//int radiusSearch(const int point_index, std::vector<int>& indices, const bool* sqr_distances, int total_points);

	/**
	* Finds all clusters in the given point cloud that are conformant to the given parameters.
	* plainPointCloud: point cloud to cluster
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


	virtual void clusterAndColor(
		const PlainPointCloud& plainPointCloud,
		ColorPointCloud& colorPointCloud,
		BoundingboxArray& clusterBoundingBoxes,
		Centroid& clusterCentroids,
		double max_cluster_distance) override;

	void segmentByDistance(
		const PlainPointCloud& plainPointCloud,
		ColorPointCloud& colorPointCloud,
		BoundingboxArray& clusterBoundingBoxes,
		Centroid& clusterCentroids) override;
};
#endif // EPHOS_EUCLIDEAN_CLUSTERING_H
