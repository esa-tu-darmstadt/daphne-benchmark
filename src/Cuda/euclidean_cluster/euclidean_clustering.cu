/**
 * Author:  Florian Stock, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2018
 * License: Apache 2.0 (see attached files)
 */
#include <iostream>
#include <fstream>
#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>

#include "euclidean_clustering.h"

// maximum allowed deviation from the reference data
#define MAX_EPS 0.001
// number of threads
#define THREADS 512


/**
 * Computes the pairwise distance indicators. Results are stored in a matrix.
 * An entry of that matrix indicates whether the distance of the two described points 
 * is less or equal to the reference distance.
 * pointcloud: points for which we need pairwise distances with size N
 * cloudSize: number of elements in the points array (N)
 * nearMap: resulting distance matrix of size N*N
 * sqrRadius: reference distance
 */
__global__ void computeInitRadiusSearch(const Point* __restrict__ pointcloud, int cloudSize, bool* __restrict__ nearMatrix, double sqrRadius)
{
	// 16 byte aligned memory accesses on 128 Bit memory interface
	int alignedCloudSize = cloudSize + (16 - cloudSize%16);
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (j >= cloudSize)
		return;

	for (int i = 0; i < cloudSize; i++)
	{
		float dx = pointcloud[i].x - pointcloud[j].x;
		float dy = pointcloud[i].y - pointcloud[j].y;
		float dz = pointcloud[i].z - pointcloud[j].z;
		//int array_index = i * alignedCloudSize + j;
		//int iDist = iLine + i;

		bool inRange = ((dx*dx + dy*dy + dz*dz) <= sqrRadius);
		// NOTE despite of many iterations and big steps in one thread we get better performance
		// by accessing continuous memory locations across a thread block
		int iNear = j + i*alignedCloudSize;
		nearMatrix[iNear] = inRange;
	}
}
/**
 * Performs radius search utilizing the precomputed nearness matrix.
 * The result is captured as boolean entries in the indices array.
 * seedQueu: points found should be near the point that belongs to this index
 * iQueueStart: index to start the search from
 * queueLength: number of points to search through
 * nearMap: result that indicates whether a point is near
 * nearMatrix: precomputed distance matrix
 * cloudSize: number of elements in the cloud
 */
__global__ void computeParallelRadiusSearch(
	const int* __restrict__ seedQueue, int iQueueStart, int queueLengths,
	bool* __restrict__ nearMap, const bool* __restrict__ nearMatrix, int cloudSize)
{
	int iSearchPoint = blockIdx.x * blockDim.x + threadIdx.x;
	// cancel superfluous threads
	if (iSearchPoint >= cloudSize)
		return;
	// NOTE our memory usage increases with an already aligned cloud size (+16bytes per line)
	int alignedCloudSize = cloudSize + (16-cloudSize%16);
	bool found = false;
	// search for the reference point
	for (int iQueue = iQueueStart; iQueue < queueLengths; iQueue++)
	{
		if (iSearchPoint == seedQueue[iQueue])
		{
			found = true;
			continue;
		}
		// using this index calculation we get accesses to apposing memory adresses within one thread block
		int iNear = iSearchPoint + seedQueue[iQueue]*alignedCloudSize;
		// with the method below we achieve better locality over multiple iterations in one thread
		//int iNear = iSearchPoint*alignedCloudSize + seedQueue[iQueue];
		if (nearMatrix[iNear])
			found = true;
	}
	// set the index array for a point that are near the reference
	nearMap[iSearchPoint] = found;

}

void euclidean_clustering::extractEuclideanClusters(
		const PlainPointCloud& plainPointCloud,
		float tolerance,
		std::vector<PointIndices> &clusters,
		unsigned int min_pts_per_cluster,
		unsigned int max_pts_per_cluster) {

	int cloudSize = plainPointCloud.size;
	// indicates the processed status for each point, initially unprocessed
	std::vector<bool> processed (cloudSize, false);
	// radius search results
	bool* nearMap = nullptr;
	cudaMallocManaged(&nearMap, cloudSize*sizeof(bool));
	// point indices of the currently grown cluster
	int* seedQueue = nullptr;
	cudaMallocManaged(&seedQueue, cloudSize*sizeof(int));
	// compute the pairwise distance matrix on the GPU
	bool* nearMatrix = nullptr;
	cudaMalloc(&nearMatrix, cloudSize*(cloudSize + 16)*sizeof(bool));
	dim3 threaddim(THREADS);
	dim3 blockdim((cloudSize + THREADS - 1)/THREADS);
	computeInitRadiusSearch<<<blockdim, threaddim>>>(
		plainPointCloud.data, cloudSize, nearMatrix, tolerance*tolerance);
	cudaDeviceSynchronize();

	// Process all points in the indices vector
	for (int i = 0; i < static_cast<int> (cloudSize); ++i)
	{
		// skip points that are already part of a cluster or were discarded
		if (processed[i])
			continue;
		// start a new cluster candidate with one element
		int queue_last_element = 0;
		seedQueue[queue_last_element++] = i;
		processed[i] = true;
		int new_elements = 1;
		// grow the cluster candidate until all near points have been found
		while (new_elements > 0)
		{
			computeParallelRadiusSearch<<<blockdim, threaddim>>>(seedQueue,
				queue_last_element - new_elements, queue_last_element,
				nearMap, nearMatrix, cloudSize);
			new_elements = 0;
			cudaDeviceSynchronize();
			// add previously unprocessed, near points to the cluster candidate
			for (size_t j = 0; j < cloudSize; ++j)
			{
				if (nearMap[j] == false)
					continue;
				if (processed[j])
					continue;
				seedQueue[queue_last_element++] = j;
				processed[j] = true;
				new_elements++;
			}
		}

		// add the cluster candidate as a new cluster if its size fits the requirements
		if (queue_last_element >= min_pts_per_cluster && queue_last_element <= max_pts_per_cluster)
		{
			PointIndices r;
			r.indices.resize (queue_last_element);
			for (size_t j = 0; j < queue_last_element; ++j)
				r.indices[j] = seedQueue[j];
			// make sure the cluster elements are sorted
			std::sort (r.indices.begin (), r.indices.end ());
			clusters.push_back (r);   // We could avoid a copy by working directly in the vector
		}
	}
	cudaFree(seedQueue);
	cudaFree(nearMatrix);
	cudaFree(nearMap);
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
	extractEuclideanClusters (plainPointCloud, static_cast<float>(tolerance), clusters, _cluster_size_min, _cluster_size_max );
	// sort from largest to smallest
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
		//ColorPointCloud current_cluster;
		std::vector<PointRGB> current_cluster;
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
		std::memcpy(colorPointCloud.data + colorPointCloud.size, current_cluster.data(), current_cluster.size()*sizeof(PointRGB));
		colorPointCloud.size += current_cluster.size();
		j++; k++;
	}
}

void euclidean_clustering::segmentByDistance(
	const PlainPointCloud& plainPointCloud,
	ColorPointCloud& colorPointCloud,
	BoundingboxArray& clusterBoundingBoxes,
	Centroid& clusterCentroids)
{
	// allocate result memory
	colorPointCloud.data = new PointRGB[plainPointCloud.size];
	colorPointCloud.capacity = plainPointCloud.size;
	colorPointCloud.size = 0;
	// find out about the segment target sizes
	PlainPointCloud cloudSegments[5] = {
		{ nullptr, 0, 0 },
		{ nullptr, 0, 0 },
		{ nullptr, 0, 0 },
		{ nullptr, 0, 0 },
		{ nullptr, 0, 0 }
	};
	//for (const Point* p = plainPointCloud.data; p < plainPointCloud.data + plainPointCloud.capacity; p++) {
	for (int i = 0; i < plainPointCloud.size; i++) {
		Point p = plainPointCloud.data[i];
		// categorize by distance from origin
		float origin_distance = p.x*p.x + p.y*p.y;
		if (origin_distance < 15*15 ) {
			cloudSegments[0].capacity += 1;
		}
		else if(origin_distance < 30*30) {
			cloudSegments[1].capacity += 1;
		}
		else if(origin_distance < 45*45) {
			cloudSegments[2].capacity += 1;
		}
		else if(origin_distance < 60*60) {
			cloudSegments[3].capacity += 1;
		} else {
			cloudSegments[4].capacity += 1;
		}
	}
	// allocate memory and distribute it to the differently sized segments
	Point* cloudSegmentStorage = nullptr;
	cudaMallocManaged(&cloudSegmentStorage, sizeof(Point)*plainPointCloud.size);
	unsigned int nextCloudSegmentStart = 0;
	for (int i = 0; i < 5; i++) {
		cloudSegments[i].data = cloudSegmentStorage + nextCloudSegmentStart;
		nextCloudSegmentStart += cloudSegments[i].capacity;
	}
	// copy points over into the segmnets
	//for (const Point* p = plainPointCloud.data; p < plainPointCloud.data + plainPointCloud.capacity; p++) {
	for (int i = 0; i < plainPointCloud.size; i++) {
		Point p = plainPointCloud.data[i];
		// categorize by distance from origin
		float origin_distance = p.x*p.x + p.y*p.y;
		if (origin_distance < 15*15 ) {
			cloudSegments[0].data[cloudSegments[0].size] = p;
			cloudSegments[0].size += 1;
		}
		else if(origin_distance < 30*30) {
			cloudSegments[1].data[cloudSegments[1].size] = p;
			cloudSegments[1].size += 1;
		}
		else if(origin_distance < 45*45) {
			cloudSegments[2].data[cloudSegments[2].size] = p;
			cloudSegments[2].size += 1;
		}
		else if(origin_distance < 60*60) {
			cloudSegments[3].data[cloudSegments[3].size] = p;
			cloudSegments[3].size += 1;
		} else {
			cloudSegments[4].data[cloudSegments[4].size] = p;
			cloudSegments[4].size += 1;
		}
	}
	// perform clustering and coloring on the individual segments
	double thresholds[5] = { 0.5, 1.1, 1.6, 2.3, 2.6 };
	for(unsigned int i=0; i<5; i++)
	{
		clusterAndColor(cloudSegments[i], colorPointCloud,
			clusterBoundingBoxes, clusterCentroids, thresholds[i]);
	}
	cudaFree(cloudSegmentStorage);
}

void euclidean_clustering::init() {
	std::cout << "init\n";
	euclidean_clustering_base::init();
	std::cout << "done\n" << std::endl;
}
void euclidean_clustering::quit() {
	euclidean_clustering_base::quit();
}
/**
 * Helper function for point comparison
 */
inline bool compareRGBPoints (const PointRGB &a, const PointRGB &b)
{
    if (a.x != b.x)
		return (a.x < b.x);
    else
	if (a.y != b.y)
	    return (a.y < b.y);
	else
	    return (a.z < b.z);
}

/**
 * Helper function for point comparison
 */
inline bool comparePoints (const PointDouble &a, const PointDouble &b)
{
	if (a.x != b.x)
		return (a.x < b.x);
	else
	if (a.y != b.y)
		return (a.y < b.y);
	else
		return (a.z < b.z);
}


/**
 * Helper function for bounding box comparison
 */
inline bool compareBBs (const Boundingbox &a, const Boundingbox &b)
{
	if (a.position.x != b.position.x)
		return (a.position.x < b.position.x);
	else
	if (a.position.y != b.position.y)
		return (a.position.y < b.position.y);
	else
		if (a.dimensions.x != b.dimensions.x)
			return (a.dimensions.x < b.dimensions.x);
		else
			return (a.dimensions.y < b.dimensions.y);
}

// set kernel used by main
euclidean_clustering a;
benchmark& myKernel = a;
