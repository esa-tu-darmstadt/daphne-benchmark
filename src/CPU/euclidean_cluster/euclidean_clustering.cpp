
/**
 * Author:  Florian Stock, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2018
 * Author:  Thilo Gabel, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2019
 * License: Apache 2.0 (see attached files)
 */
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <cfloat>

#include "euclidean_clustering.h"

void euclidean_clustering::initRadiusSearch(const PlainPointCloud& cloud, bool**  sqr_distances, float radius)
{
	int n = cloud.size;
	float radius_sqr = radius * radius;
	*sqr_distances = (bool*) malloc(n * n * sizeof(bool));
	for (int j = 0; j < n; j++)
		for (int i = 0; i < n; i++)
		{
			float dx = cloud.data[i].x - cloud.data[j].x;
			float dy = cloud.data[i].y - cloud.data[j].y;
			float dz = cloud.data[i].z - cloud.data[j].z;
			float sqr_distance = dx*dx + dy*dy + dz*dz;
			(*sqr_distances)[j*n+i] = sqr_distance <= radius_sqr;
		}
}
int euclidean_clustering::radiusSearch(
	const int point_index, std::vector<int>& indices, const bool* sqr_distances, int total_points)
{
	indices.clear();
	for (int i = 0; i < point_index; i++) {
		if (sqr_distances[point_index*total_points + i]) {
			indices.push_back(i);
		}
	}
	for (int i = point_index + 1; i < total_points; i++) {
		if (sqr_distances[point_index*total_points + i]) {
			indices.push_back(i);
		}
	}
	return indices.size();
}

void euclidean_clustering::extractEuclideanClusters (
	const PlainPointCloud& plainPointCloud,
	float tolerance,
	std::vector<PointIndices> &clusters,
	unsigned int min_pts_per_cluster, 
	unsigned int max_pts_per_cluster)
{
	int nn_start_idx = 0;
	int cloudSize = plainPointCloud.size;

	// indicates the processed status for each point
	std::vector<bool> processed (cloudSize, false);
	// temporary radius search results
	std::vector<int> nn_indices;
	// precompute the distance matrix
	bool *sqr_distances;
	initRadiusSearch(plainPointCloud, &sqr_distances, tolerance);

	// iterate for all points in the cloud
	for (int i = 0; i < cloudSize; ++i)
	{
		// ignore if already tested
		if (processed[i])
			continue;
		// begin a cluster candidate with one item
		std::vector<int> seed_queue;
		int sq_idx = 0;
		seed_queue.push_back(i);
		processed[i] = true;

		// grow the cluster candidate until all items have been searched through
		while (sq_idx < seed_queue.size())
		{
			int ret = radiusSearch(seed_queue[sq_idx], nn_indices, sqr_distances, cloudSize);
			if (!ret)
			{
				sq_idx++;
				continue;
			}
			// add indices of near points to the cluster candidate
			for (size_t j = nn_start_idx; j < nn_indices.size (); ++j)             // can't assume sorted (default isn't!)
			{
				if (nn_indices[j] == -1 || processed[nn_indices[j]])        // Has this point been processed before ?
					continue;
				seed_queue.push_back (nn_indices[j]);
				processed[nn_indices[j]] = true;
			}
			sq_idx++;
		}

		// add cluster candidate of fitting size to the resulting clusters
		if (seed_queue.size() >= min_pts_per_cluster && seed_queue.size() <= max_pts_per_cluster)
		{
			PointIndices r;
			r.indices.resize(seed_queue.size ());
			for (size_t j = 0; j < seed_queue.size (); ++j)
				r.indices[j] = seed_queue[j];
			std::sort (r.indices.begin (), r.indices.end ());
			clusters.push_back(r);
		}
	}
	free(sqr_distances);
}

/**
 * Helper function that compares cluster sizes.
 */
inline bool comparePointClusters (const PointIndices &a, const PointIndices &b)
{
	return (a.indices.size () < b.indices.size ());
}

/**
 * Computes euclidean clustering and sorts the resulting clusters.
 */
void euclidean_clustering::extract(
	const PlainPointCloud& plainPointCloud,
	std::vector<PointIndices> &clusters, 
	double tolerance)
{
	if (plainPointCloud.size == 0)
	{
	    clusters.clear ();
	    return;
	}
	extractEuclideanClusters (
		plainPointCloud, static_cast<float>(tolerance),
		clusters, _cluster_size_min, _cluster_size_max);
	// sort by number of elements
	std::sort (clusters.rbegin (), clusters.rend (), comparePointClusters);
}

void euclidean_clustering::clusterAndColor(
	const PlainPointCloud& plainPointCloud,
	ColorPointCloud& colorPointCloud,
	BoundingboxArray& clusterBoundingBoxes,
	Centroid& clusterCentroids,
	double max_cluster_distance=0.5)
{
	std::vector<PointIndices> cluster_indices;
	extract(plainPointCloud,
		cluster_indices,
		max_cluster_distance);

	// assign colors
	int j = 0;
	unsigned int k = 0;
	for (auto it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
	{
		//ColorPointCloud* current_cluster = new ColorPointCloud;//coord + color cluster
		ColorPointCloud current_cluster;
		// assign color to each cluster
		PointDouble centroid = {0.0, 0.0, 0.0};
		for (auto pit = it->indices.begin(); pit != it->indices.end(); ++pit)
		{
			// fill new colored cluster point by point
			PointRGB p;
			p.x = (plainPointCloud.data)[*pit].x;
			p.y = (plainPointCloud.data)[*pit].y;
			p.z = (plainPointCloud.data)[*pit].z;
			p.r = 10;
			p.g = 20;
			p.b = 30;
			centroid.x += (plainPointCloud.data)[*pit].x;
			centroid.y += (plainPointCloud.data)[*pit].y;
			centroid.z += (plainPointCloud.data)[*pit].z;

			current_cluster.push_back(p);
		}

		centroid.x /= it->indices.size();
		centroid.y /= it->indices.size();
		centroid.z /= it->indices.size();

		// get extends
		float min_x=std::numeric_limits<float>::max();float max_x=-std::numeric_limits<float>::max();
		float min_y=std::numeric_limits<float>::max();float max_y=-std::numeric_limits<float>::max();
		float min_z=std::numeric_limits<float>::max();float max_z=-std::numeric_limits<float>::max();
		for(unsigned int i=0; i<current_cluster.size();i++)
		{
			if(current_cluster[i].x<min_x)  min_x = current_cluster[i].x;
			if(current_cluster[i].y<min_y)  min_y = current_cluster[i].y;
			if(current_cluster[i].z<min_z)  min_z = current_cluster[i].z;
			if(current_cluster[i].x>max_x)  max_x = current_cluster[i].x;
			if(current_cluster[i].y>max_y)  max_y = current_cluster[i].y;
			if(current_cluster[i].z>max_z)  max_z = current_cluster[i].z;
		}
		float l = max_x - min_x;
		float w = max_y - min_y;
		float h = max_z - min_z;
		// create a bounding box from cluster extends
		Boundingbox bounding_box;
		bounding_box.position.x = min_x + l/2;
		bounding_box.position.y = min_y + w/2;
		bounding_box.position.z = min_z + h/2;
		bounding_box.dimensions.x = ((l<0)?-1*l:l);
		bounding_box.dimensions.y = ((w<0)?-1*w:w);
		bounding_box.dimensions.z = ((h<0)?-1*h:h);

		double rz = 0;
		// estimate pose
		if (_pose_estimation) 
		{
			std::vector<Point2D> inner_points;
			for (unsigned int i=0; i < current_cluster.size(); i++)
			{
				Point2D ip;
				ip.x = (current_cluster[i].x + fabs(min_x))*8;
				ip.y = (current_cluster[i].y + fabs(min_y))*8;
				inner_points.push_back(ip);
			}

			if (inner_points.size() > 0)
			{
				rz = minAreaRectAngle(inner_points) * PI / 180.0;
			}
		}

		// quaternion for rotation stored in bounding box
		double halfYaw = rz * 0.5;  
		double cosYaw = cos(halfYaw);
		double sinYaw = sin(halfYaw);
		bounding_box.orientation.x = 0.0;
		bounding_box.orientation.y = 0.0;
		bounding_box.orientation.z = sinYaw;
		bounding_box.orientation.w = cosYaw;

		if (bounding_box.dimensions.x >0 && bounding_box.dimensions.y >0 && bounding_box.dimensions.z > 0 &&
			bounding_box.dimensions.x < 15 && bounding_box.dimensions.y >0 && bounding_box.dimensions.y < 15 &&
			max_z > -1.5 && min_z > -1.5 && min_z < 1.0 )
		{
			clusterBoundingBoxes.boxes.push_back(bounding_box);
			clusterCentroids.points.push_back(centroid);
		}
		colorPointCloud.insert(colorPointCloud.end(), current_cluster.begin(), current_cluster.end());
		j++; k++;
	}

}


void euclidean_clustering::init() {
	std::cout << "init\n";
	euclidean_clustering_base::init();
	std::cout << "done" << std::endl;
}
void euclidean_clustering::quit() {
	euclidean_clustering_base::quit();
}
// create the benchmark to be run from main
euclidean_clustering a;
benchmark& myKernel = a;