
/**
 * Author:  Florian Stock, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2018
 * Author:  Leonardo Solis, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2018
 * Author:  Thilo Gabel, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2019 - 2020
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

#include "common/benchmark.h"
//#include "datatypes.h"
#include "euclidean_clustering.h"
//#include "common/compute_tools.h"
#include "kernel/kernel.h"

euclidean_clustering::euclidean_clustering() :
	euclidean_clustering_base(),
	computeEnv(),
	distanceMatrixKernel(),
	radiusSearchKernel(),
	seedQueueBuffer(),
	processedBuffer(),
	distanceBuffer(),
	pointCloudBuffer(),
	seedQueueLengthBuffer(),
	maxSeedQueueLength(0)
{}

euclidean_clustering::~euclidean_clustering() {}

void euclidean_clustering::prepare_compute_buffers(const PlainPointCloud& pointCloud, int cloudSize) {
	if (maxSeedQueueLength >= cloudSize) {
		return;
	}
	int alignedCloudSize;
	int alignTo = EPHOS_KERNEL_DISTANCES_PER_PACKET*EPHOS_KERNEL_DISTANCE_PACKETS_PER_ITEM;
	if (cloudSize%alignTo == 0) {
		alignedCloudSize = cloudSize;
	} else {
		alignedCloudSize = (cloudSize/alignTo + 1)*alignTo;
	}
	maxSeedQueueLength = alignedCloudSize;
	int distanceLineLength = alignedCloudSize/EPHOS_KERNEL_DISTANCES_PER_PACKET;

	pointCloudBuffer = cl::Buffer(computeEnv.context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY, sizeof(Point)*alignedCloudSize);
	seedQueueBuffer = cl::Buffer(computeEnv.context, CL_MEM_READ_WRITE, sizeof(int)*cloudSize);
	distanceBuffer = cl::Buffer(computeEnv.context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, sizeof(DistancePacket)*cloudSize*distanceLineLength);
	processedBuffer = cl::Buffer(computeEnv.context, CL_MEM_READ_WRITE | CL_MEM_HOST_WRITE_ONLY, sizeof(bool)*alignedCloudSize);
	seedQueueLengthBuffer = cl::Buffer(computeEnv.context, CL_MEM_READ_WRITE, sizeof(int));

}

void euclidean_clustering::extractEuclideanClusters(
	const PlainPointCloud& plainPointCloud,
		float tolerance,
		std::vector<PointIndices> &clusters,
		unsigned int min_pts_per_cluster,
		unsigned int max_pts_per_cluster)
{
	cl_int err;
	// calculate cloud size aligned to distances per item
	int cloudSize = plainPointCloud.size;
	int alignedCloudSize;
	int alignTo = EPHOS_KERNEL_DISTANCES_PER_PACKET*EPHOS_KERNEL_DISTANCE_PACKETS_PER_ITEM;
	if ((cloudSize%alignTo) == 0) {
		alignedCloudSize = cloudSize;
	} else {
		alignedCloudSize = (cloudSize/alignTo + 1)*alignTo;
	}
	int distanceLineLength = alignedCloudSize/EPHOS_KERNEL_DISTANCES_PER_PACKET;

	// move point cloud to device
	// due to alignment a tail of the buffer can remain undefined
	// the entries in the processed buffer are set accordingly in the first kernel
	// so that the distance matrix entries for these points do not matter
#ifdef EPHOS_ZERO_COPY
	Point* cloudStorage = (Point *) computeEnv.cmdqueue.enqueueMapBuffer(pointCloudBuffer, CL_TRUE,
		CL_MAP_WRITE_INVALIDATE_REGION, 0, sizeof(Point)*cloudSize);
	std::memcpy(cloudStorage, plainPointCloud.data, sizeof(Point)*cloudSize);
	computeEnv.cmdqueue.enqueueUnmapMemObject(pointCloudBuffer, cloudStorage);

#else // !EPHOS_ZERO_COPY

// 	if (alignedCloudSize > cloudSize) {
// 		float farAway = FLT_MAX*0.5f;
// 		computeEnv.cmdqueue.enqueueFillBuffer(pointCloudBuffer, farAway,
// 			sizeof(Point)*cloudSize, sizeof(Point)*(alignedCloudSize - cloudSize));
// 	}

// 	if (alignedCloudSize > cloudSize) {
// 		std::vector<float> farAway(sizeof(Point)/sizeof(float)*(alignedCloudSize - cloudSize), FLT_MAX*0.5f);
// 		computeEnv.cmdqueue.enqueueWriteBuffer(pointCloudBuffer, CL_FALSE,
// 			sizeof(Point)*cloudSize, sizeof(Point)*(alignedCloudSize - cloudSize), farAway.data());
// 	}
	computeEnv.cmdqueue.enqueueWriteBuffer(pointCloudBuffer, CL_FALSE,
		0, sizeof(Point)*cloudSize, plainPointCloud.data);
#endif // !EPHOS_ZERO_COPY


	// calculate global range aligned to work group size
	int globalRange = alignedCloudSize/EPHOS_KERNEL_DISTANCE_PACKETS_PER_ITEM;
	int workGroupNo;
	if (globalRange%EPHOS_KERNEL_WORK_GROUP_SIZE == 0) {
		workGroupNo = globalRange/EPHOS_KERNEL_WORK_GROUP_SIZE;
	} else {
		workGroupNo = globalRange/EPHOS_KERNEL_WORK_GROUP_SIZE + 1;
		globalRange = workGroupNo*EPHOS_KERNEL_WORK_GROUP_SIZE;
	}
	cl::NDRange offsetNDRange(0);
	cl::NDRange localNDRange (EPHOS_KERNEL_WORK_GROUP_SIZE);
	cl::NDRange globalNDRange(globalRange);

	RadiusSearchInfo searchInfo = {
		(double)(tolerance*tolerance),
		cloudSize,
		alignedCloudSize,
		distanceLineLength,
		0, // queue start
		1 // queue length
	};
	// call the initialization kernel
	distanceMatrixKernel.setArg(0, pointCloudBuffer);
	distanceMatrixKernel.setArg(1, distanceBuffer);
	distanceMatrixKernel.setArg(2, processedBuffer);
	distanceMatrixKernel.setArg(3, searchInfo);

 	computeEnv.cmdqueue.enqueueNDRangeKernel(
 		distanceMatrixKernel, offsetNDRange, globalNDRange, localNDRange);
	// raidus search progress indicators
	std::vector<Processed> processed(alignedCloudSize, 0);
	//bool* processed = new bool[cloudSize];
	std::memset(processed.data() + cloudSize, 1, sizeof(Processed)*(alignedCloudSize - cloudSize));
//	computeEnv.cmdqueue.enqueueWriteBuffer(processedBuffer, CL_FALSE,
//		0, sizeof(Processed)*alignedCloudSize, processed.data());
	radiusSearchKernel.setArg(0, seedQueueBuffer);
	radiusSearchKernel.setArg(1, distanceBuffer);
	radiusSearchKernel.setArg(2, processedBuffer);
	radiusSearchKernel.setArg(3, seedQueueLengthBuffer);
	// Process all points in the indices vector
	for (int i = 0; i < cloudSize; ++i)
	{
		// skip elements that have already been looked at
		if (processed[i])
			continue;
		// begin a new candidate with one element
		processed[i] = true;
		int staticCandidateNo = 0;
		int nextCandidateNo = 1;
		Processed proc = 0x1;
		// TODO: enable when done testing
		computeEnv.cmdqueue.enqueueWriteBuffer(seedQueueBuffer, CL_FALSE,
			0, sizeof(int), &i);
		computeEnv.cmdqueue.enqueueWriteBuffer(processedBuffer, CL_FALSE,
			sizeof(Processed)*i, sizeof(Processed), &proc);
		computeEnv.cmdqueue.enqueueWriteBuffer(seedQueueLengthBuffer, CL_FALSE,
			0, sizeof(int), &nextCandidateNo);
		// grow the candidate until convergence
		while (nextCandidateNo > staticCandidateNo)
		{
			searchInfo.queueStartIndex = staticCandidateNo;
			searchInfo.staticQueueSize = nextCandidateNo;
			// call the radius search kernel
			radiusSearchKernel.setArg(4, searchInfo);
			computeEnv.cmdqueue.enqueueNDRangeKernel(
				radiusSearchKernel, offsetNDRange, globalNDRange, localNDRange);
			// update counters
			staticCandidateNo = nextCandidateNo;
			computeEnv.cmdqueue.enqueueReadBuffer(seedQueueLengthBuffer, CL_TRUE,
				0, sizeof(int), &nextCandidateNo);

		}
		staticCandidateNo = nextCandidateNo;
		// add the cluster candidate if it is inside satisfactory size bounds
		if (nextCandidateNo >= min_pts_per_cluster && nextCandidateNo <= max_pts_per_cluster)
		{
			// store the cluster
			int iCluster = clusters.size();
			clusters.resize(iCluster + 1);
			PointIndices& cluster = clusters[iCluster];
			cluster.indices.resize(nextCandidateNo);
			computeEnv.cmdqueue.enqueueReadBuffer(seedQueueBuffer, CL_TRUE,
				0, sizeof(int)*nextCandidateNo, cluster.indices.data());
			std::sort(cluster.indices.begin(), cluster.indices.end());
			for (int j = 1; j < nextCandidateNo; j++) {
				processed[cluster.indices[j]] = 0x1;
			}
		} else if (nextCandidateNo > 1) {
			// mark all except the starting element as processsed
			int* candidateStorage = (int *) computeEnv.cmdqueue.enqueueMapBuffer(seedQueueBuffer, CL_TRUE, CL_MAP_READ,
				sizeof(int), sizeof(int)*(nextCandidateNo - 1));
			for (int j = 0; j < nextCandidateNo - 1; j++) {
				processed[candidateStorage[j]] = 0x1;
			}
			computeEnv.cmdqueue.enqueueUnmapMemObject(seedQueueBuffer, candidateStorage);
		}
	}
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

// void euclidean_clustering::clusterAndColor(
// 	const PlainPointCloud& plainPointCloud,
// 	ColorPointCloud& colorPointCloud,
// 	BoundingboxArray& clusterBoundingBoxes,
// 	Centroid& clusterCentroids,
// 	double max_cluster_distance=0.5)
// {
// 	std::vector<PointIndices> cluster_indices;
// 	extract(plainPointCloud,
// 		cluster_indices,
// 		max_cluster_distance);
//
// 	// assign colors
// 	int j = 0;
// 	unsigned int k = 0;
// 	for (auto it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
// 	{
// 		//ColorPointCloud* current_cluster = new ColorPointCloud;//coord + color cluster
// 		ColorPointCloud current_cluster;
// 		// assign color to each cluster
// 		PointDouble centroid = {0.0, 0.0, 0.0};
// 		for (auto pit = it->indices.begin(); pit != it->indices.end(); ++pit)
// 		{
// 			// fill new colored cluster point by point
// 			PointRGB p;
// 			p.x = (plainPointCloud)[*pit].x;
// 			p.y = (plainPointCloud)[*pit].y;
// 			p.z = (plainPointCloud)[*pit].z;
// 			p.r = 10;
// 			p.g = 20;
// 			p.b = 30;
// 			centroid.x += (plainPointCloud)[*pit].x;
// 			centroid.y += (plainPointCloud)[*pit].y;
// 			centroid.z += (plainPointCloud)[*pit].z;
//
// 			current_cluster.push_back(p);
// 		}
//
// 		centroid.x /= it->indices.size();
// 		centroid.y /= it->indices.size();
// 		centroid.z /= it->indices.size();
//
// 		// get extends
// 		float min_x=std::numeric_limits<float>::max();float max_x=-std::numeric_limits<float>::max();
// 		float min_y=std::numeric_limits<float>::max();float max_y=-std::numeric_limits<float>::max();
// 		float min_z=std::numeric_limits<float>::max();float max_z=-std::numeric_limits<float>::max();
// 		for(unsigned int i=0; i<current_cluster.size();i++)
// 		{
// 			if(current_cluster[i].x<min_x)  min_x = current_cluster[i].x;
// 			if(current_cluster[i].y<min_y)  min_y = current_cluster[i].y;
// 			if(current_cluster[i].z<min_z)  min_z = current_cluster[i].z;
// 			if(current_cluster[i].x>max_x)  max_x = current_cluster[i].x;
// 			if(current_cluster[i].y>max_y)  max_y = current_cluster[i].y;
// 			if(current_cluster[i].z>max_z)  max_z = current_cluster[i].z;
// 		}
// 		float l = max_x - min_x;
// 		float w = max_y - min_y;
// 		float h = max_z - min_z;
// 		// create a bounding box from cluster extends
// 		Boundingbox bounding_box;
// 		bounding_box.position.x = min_x + l/2;
// 		bounding_box.position.y = min_y + w/2;
// 		bounding_box.position.z = min_z + h/2;
// 		bounding_box.dimensions.x = ((l<0)?-1*l:l);
// 		bounding_box.dimensions.y = ((w<0)?-1*w:w);
// 		bounding_box.dimensions.z = ((h<0)?-1*h:h);
//
// 		double rz = 0;
// 		// estimate pose
// 		if (_pose_estimation)
// 		{
// 			std::vector<Point2D> inner_points;
// 			for (unsigned int i=0; i < current_cluster.size(); i++)
// 			{
// 				Point2D ip;
// 				ip.x = (current_cluster[i].x + fabs(min_x))*8;
// 				ip.y = (current_cluster[i].y + fabs(min_y))*8;
// 				inner_points.push_back(ip);
// 			}
//
// 			if (inner_points.size() > 0)
// 			{
// 				rz = minAreaRectAngle(inner_points) * PI / 180.0;
// 			}
// 		}
//
// 		// quaternion for rotation stored in bounding box
// 		double halfYaw = rz * 0.5;
// 		double cosYaw = cos(halfYaw);
// 		double sinYaw = sin(halfYaw);
// 		bounding_box.orientation.x = 0.0;
// 		bounding_box.orientation.y = 0.0;
// 		bounding_box.orientation.z = sinYaw;
// 		bounding_box.orientation.w = cosYaw;
//
// 		if (bounding_box.dimensions.x >0 && bounding_box.dimensions.y >0 && bounding_box.dimensions.z > 0 &&
// 			bounding_box.dimensions.x < 15 && bounding_box.dimensions.y >0 && bounding_box.dimensions.y < 15 &&
// 			max_z > -1.5 && min_z > -1.5 && min_z < 1.0 )
// 		{
// 			clusterBoundingBoxes.boxes.push_back(bounding_box);
// 			clusterCentroids.points.push_back(centroid);
// 		}
// 		colorPointCloud.insert(colorPointCloud.end(), current_cluster.begin(), current_cluster.end());
// 		j++; k++;
// 	}
//
// }
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
		//colorPointCloud.insert(colorPointCloud.end(), current_cluster.begin(), current_cluster.end());
		//std::memcpy(colorPointCloud.data + colorPointCloud.size, current_cluster.data(), current_cluster.size());
		// TODO replace
		for (int iPoint = 0; iPoint < current_cluster.size(); iPoint++) {
			colorPointCloud.data[colorPointCloud.size + iPoint] = current_cluster[iPoint];
		}
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
	Point* cloudSegmentStorage = new Point[plainPointCloud.size];
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
	// preparation for kernel calls
	int iBigSegment = 0;
	int bigSegmentSize = cloudSegments[0].size;
	for (int s = 1; s < 5; s++) {
		if (cloudSegments[s].size > bigSegmentSize) {
			iBigSegment = s;
			bigSegmentSize = cloudSegments[s].size;
		}
	}
	prepare_compute_buffers(cloudSegments[iBigSegment], bigSegmentSize);
	// perform clustering and coloring on the individual segments
	double thresholds[5] = { 0.5, 1.1, 1.6, 2.3, 2.6 };
	for(unsigned int i=0; i<5; i++)
	{
		clusterAndColor(cloudSegments[i], colorPointCloud,
			clusterBoundingBoxes, clusterCentroids, thresholds[i]);
	}
	delete[] cloudSegmentStorage;
}
// void euclidean_clustering::segmentByDistance(
// 	const PlainPointCloud& plainPointCloud,
// 	ColorPointCloud& colorPointCloud,
// 	BoundingboxArray& clusterBoundingBoxes,
// 	Centroid& clusterCentroids)
// {
// 	PlainPointCloud cloud_segments_array[5];
// 	double thresholds[5] = {0.5, 1.1, 1.6, 2.3, 2.6f};
//
// 	for (const Point& p : plainPointCloud) {
//
// 		// categorize by distance from origin
// 		float origin_distance = p.x*p.x + p.y*p.y;
// 		if (origin_distance < 15*15 ) {
// 			cloud_segments_array[0].push_back(p);
// 		}
// 		else if(origin_distance < 30*30) {
// 			cloud_segments_array[1].push_back(p);
// 		}
// 		else if(origin_distance < 45*45) {
// 			cloud_segments_array[2].push_back(p);
// 		}
// 		else if(origin_distance < 60*60) {
// 			cloud_segments_array[3].push_back(p);
// 		} else {
// 			cloud_segments_array[4].push_back(p);
// 		}
// 	}
// 	// find biggest segment and prepare compute resources
// 	int iBigSegment = 0;
// 	int bigSegmentSize = cloud_segments_array[0].size();
// 	for (int s = 1; s < 5; s++) {
// 		if (cloud_segments_array[s].size() > bigSegmentSize) {
// 			iBigSegment = s;
// 			bigSegmentSize = cloud_segments_array[s].size();
// 		}
// 	}
// 	prepare_compute_buffers(cloud_segments_array[iBigSegment], bigSegmentSize);
// 	// perform clustering and coloring on the individual categories
// 	for(unsigned int i=0; i<5; i++)
// 	{
// 		clusterAndColor(cloud_segments_array[i], colorPointCloud, clusterBoundingBoxes, clusterCentroids, thresholds[i]);
// 	}
// }

void euclidean_clustering::init() {
	std::cout << "init\n";
	euclidean_clustering_base::init();
	try {
		std::vector<std::vector<std::string>> requiredExtensions = { {"cl_khr_fp64", "cl_amd_fp64"} };
		computeEnv = ComputeTools::find_compute_platform(EPHOS_PLATFORM_HINT_S, EPHOS_DEVICE_HINT_S,
			EPHOS_DEVICE_TYPE_S, requiredExtensions);
		std::cout << "OpenCL device: " << computeEnv.device.getInfo<CL_DEVICE_NAME>() << std::endl;
	} catch (std::logic_error& e) {
		std::cerr << e.what() << std::endl;
		exit(EXIT_FAILURE);
	}
	// build program from stringified source code
	std::string sourceCode = radius_search_ocl_kernel_source;
	std::vector<cl::Kernel> kernels;
	try {
		std::string sBuildOptions =
#ifdef EPHOS_KERNEL_ATOMICS
		"-DEPHOS_ATOMICS "
#endif
#ifdef EPHOS_KERNEL_LINE_PROCESSING
		"-DEPHOS_LINE_PROCESSING "
#endif
#ifdef EPHOS_KERNEL_DISTANCES_PER_PACKET
		"-DEPHOS_DISTANCES_PER_PACKET=" STRINGIFY(EPHOS_KERNEL_DISTANCES_PER_PACKET) " "
#endif
#ifdef EPHOS_KERNEL_DISTANCE_PACKETS_PER_ITEM
		"-DEPHOS_DISTANCE_PACKETS_PER_ITEM=" STRINGIFY(EPHOS_KERNEL_DISTANCE_PACKETS_PER_ITEM) " "
#endif
		"";
		std::vector<std::string> kernelNames {
			"distanceMatrix",
			"radiusSearch"
		};
		cl::Program program = ComputeTools::build_program(computeEnv, sourceCode, sBuildOptions,
			kernelNames, kernels);
	} catch (std::logic_error& e) {
		std::cerr << e.what() << std::endl;
		exit(EXIT_FAILURE);
	}
 	distanceMatrixKernel = kernels[0];
 	radiusSearchKernel = kernels[1];
	maxSeedQueueLength = -1;

	std::cout << "done" << std::endl;
}
void euclidean_clustering::quit() {
	euclidean_clustering_base::quit();
	// free compute resources
	computeEnv.cmdqueue = cl::CommandQueue();
	computeEnv.context = cl::Context();
	computeEnv.device = cl::Device();

}
euclidean_clustering a;
benchmark& myKernel = a;