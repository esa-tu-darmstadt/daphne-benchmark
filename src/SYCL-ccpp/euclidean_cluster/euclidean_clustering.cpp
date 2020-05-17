
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

#include <SYCL/sycl.hpp>

#include "common/benchmark.h"
#include "datatypes.h"
#include "euclidean_clustering.h"
#include "common/compute_tools.h"

euclidean_clustering::euclidean_clustering() :
	plainPointCloud(),
	colorPointCloud(),
	clusterBoundingBoxes(),
	clusterCentroids(),
	read_testcases(0),
	input_file(),
	output_file(),
	datagen_file(),
	error_so_far(false),
	max_delta(0),
	computeEnv()
{}

euclidean_clustering::~euclidean_clustering() {}

void euclidean_clustering::rotatingCalipers( const Point2D* points, int n, float* out )
{
	float minarea = std::numeric_limits<float>::max();
	float max_dist = 0;
	char buffer[32] = {};
	int i, k;
	float* abuf = (float*)alloca(n * 3 * sizeof(float));
	float* inv_vect_length = abuf;
	Point2D* vect = (Point2D*)(inv_vect_length + n);
	int left = 0, bottom = 0, right = 0, top = 0;
	int seq[4] = { -1, -1, -1, -1 };
	float orientation = 0;
	float base_a;
	float base_b = 0;

	float left_x, right_x, top_y, bottom_y;
	Point2D pt0 = points[0];
	left_x = right_x = pt0.x;
	top_y = bottom_y = pt0.y;

	for( i = 0; i < n; i++ )
	{
		double dx, dy;
		if( pt0.x < left_x )
			left_x = pt0.x, left = i;
		if( pt0.x > right_x )
			right_x = pt0.x, right = i;
		if( pt0.y > top_y )
			top_y = pt0.y, top = i;
		if( pt0.y < bottom_y )
			bottom_y = pt0.y, bottom = i;
		Point2D pt = points[(i+1) & (i+1 < n ? -1 : 0)];
		dx = pt.x - pt0.x;
		dy = pt.y - pt0.y;
		vect[i].x = (float)dx;
		vect[i].y = (float)dy;
		inv_vect_length[i] = (float)(1./std::sqrt(dx*dx + dy*dy));
		pt0 = pt;
	}

	// find convex hull orientation
	{
		double ax = vect[n-1].x;
		double ay = vect[n-1].y;
		for( i = 0; i < n; i++ )
		{
			double bx = vect[i].x;
			double by = vect[i].y;
			double convexity = ax * by - ay * bx;

			if( convexity != 0 )
			{
				orientation = (convexity > 0) ? 1.f : (-1.f);
				break;
			}
			ax = bx;
			ay = by;
		}
		// orientation should be 0 at this point
	}
	base_a = orientation;
	// init caliper position
	seq[0] = bottom;
	seq[1] = right;
	seq[2] = top;
	seq[3] = left;
	// main loop
	// evaluate angles and rotate calipers
	// all of edges will be checked while rotating calipers by 90 degrees
	for( k = 0; k < n; k++ )
	{
		// compute cosine of angle between calipers side and polygon edge
		// dp - dot product
		float dp[4] = {
			+base_a * vect[seq[0]].x + base_b * vect[seq[0]].y,
			-base_b * vect[seq[1]].x + base_a * vect[seq[1]].y,
			-base_a * vect[seq[2]].x - base_b * vect[seq[2]].y,
			+base_b * vect[seq[3]].x - base_a * vect[seq[3]].y,
		};
		float maxcos = dp[0] * inv_vect_length[seq[0]];
		// number of calipers edges, that has minimal angle with edge
		int main_element = 0;
		// choose minimal angle
		for ( i = 1; i < 4; ++i )
		{
			float cosalpha = dp[i] * inv_vect_length[seq[i]];
			if (cosalpha > maxcos)
			{
				main_element = i;
				maxcos = cosalpha;
			}
		}
		// rotate calipers
		{
			//get next base
			int pindex = seq[main_element];
			float lead_x = vect[pindex].x*inv_vect_length[pindex];
			float lead_y = vect[pindex].y*inv_vect_length[pindex];
			switch( main_element )
			{
			case 0:
				base_a = lead_x;
				base_b = lead_y;
				break;
			case 1:
				base_a = lead_y;
				base_b = -lead_x;
				break;
			case 2:
				base_a = -lead_x;
				base_b = -lead_y;
				break;
			case 3:
				base_a = -lead_y;
				base_b = lead_x;
				break;
			default:
				throw std::logic_error("main_element should be 0, 1, 2 or 3");
			}
		}
		// change base point of main edge
		seq[main_element] += 1;
		seq[main_element] = (seq[main_element] == n) ? 0 : seq[main_element];

		// find area of rectangle
		{
			float height;
			float area;
			// find left-right vector
			float dx = points[seq[1]].x - points[seq[3]].x;
			float dy = points[seq[1]].y - points[seq[3]].y;
			// dot(d, base)
			float width = dx * base_a + dy * base_b;
			// find vector left-right
			dx = points[seq[2]].x - points[seq[0]].x;
			dy = points[seq[2]].y - points[seq[0]].y;
			// dot(inv(d, b));
			height = -dx * base_b + dy * base_a;
		
			area = width * height;
			if( area <= minarea )
			{
				float *buf = (float *) buffer;
		
				minarea = area;
				// leftmost point
				((int *) buf)[0] = seq[3];
				buf[1] = base_a;
				buf[2] = width;
				buf[3] = base_b;
				buf[4] = height;
				// bottom point
				((int *) buf)[5] = seq[0];
				buf[6] = area;
			}
		}
	}

	float *buf = (float *) buffer;

	float A1 = buf[1];
	float B1 = buf[3];

	float A2 = -buf[3];
	float B2 = buf[1];

	float C1 = A1 * points[((int *) buf)[0]].x + points[((int *) buf)[0]].y * B1;
	float C2 = A2 * points[((int *) buf)[5]].x + points[((int *) buf)[5]].y * B2;

	float idet = 1.f / (A1 * B2 - A2 * B1);

	float px = (C1 * B2 - C2 * B1) * idet;
	float py = (A1 * C2 - A2 * C1) * idet;

	out[0] = px;
	out[1] = py;

	out[2] = A1 * buf[2];
	out[3] = B1 * buf[2];

	out[4] = A2 * buf[4];
	out[5] = B2 * buf[4];
}

int euclidean_clustering::sklansky(Point2D** array, int start, int end, int* stack, int nsign, int sign2)
{
	int incr = end > start ? 1 : -1;
	// prepare first triangle
	int pprev = start, pcur = pprev + incr, pnext = pcur + incr;
	int stacksize = 3;

	if (start == end ||
		(array[start]->x == array[end]->x &&
		array[start]->y == array[end]->y))
	{
		stack[0] = start;
		return 1;
	}
	stack[0] = pprev;
	stack[1] = pcur;
	stack[2] = pnext;

	end += incr;

	while( pnext != end )
	{
		// check the angles p1,p2,p3
		float cury = array[pcur]->y;
		float nexty = array[pnext]->y;
		float by = nexty - cury;

		if((by > 0) - (by < 0) != nsign )
		{
			float ax = array[pcur]->x - array[pprev]->x;
			float bx = array[pnext]->x - array[pcur]->x;
			float ay = cury - array[pprev]->y;
			float convexity = ay*bx - ax*by; // convexity > 0 -> convex angle

			if(((convexity > 0) - (convexity < 0)) == sign2 && (ax != 0 || ay != 0) )
			{
				pprev = pcur;
				pcur = pnext;
				pnext += incr;
				stack[stacksize] = pnext;
				stacksize++;
			}
			else
			{
				if( pprev == start )
				{
					pcur = pnext;
					stack[1] = pcur;
					pnext += incr;
					stack[2] = pnext;
				}
				else
				{
					stack[stacksize-2] = pnext;
					pcur = pprev;
					pprev = stack[stacksize-4];
					stacksize--;
				}
			}
		}
		else
		{
			pnext += incr;
			stack[stacksize-1] = pnext;
		}
	}
	return --stacksize;
}


/**
 * Helper function for point comparison
 */
bool comparePoint2D(const Point2D* p1, const Point2D* p2) {
	return p1->x < p2->x || (p1->x == p2->x && p1->y < p2->y);
}

void euclidean_clustering::convexHull(
	std::vector<Point2D> _points, std::vector<Point2D>&  _hull, bool clockwise, bool returnPoints )
{
	int i, total = _points.size(), nout = 0;
	int miny_ind = 0, maxy_ind = 0;
	// test for empty input
	if( total == 0 )
	{
		_hull.clear();
		return;
	}

	Point2D** _pointer = (Point2D**)alloca(total * sizeof(Point2D*));
	int* _stack = (int*)alloca((total +2) * sizeof(int));
	int* _hullbuf= (int*)alloca(total * sizeof(int));
	Point2D** pointer = _pointer;
	Point2D** pointerf = (Point2D**)pointer;
	Point2D* data0 = _points.data();
	int* stack = _stack;
	int* hullbuf = _hullbuf;

	for( i = 0; i < total; i++ )
		pointer[i] = &data0[i];

	// sort the point set by x-coordinate, find min and max y
	std::sort(pointerf, pointerf + total, comparePoint2D);
	for( i = 1; i < total; i++ )
		{
			float y = pointerf[i]->y;
			if( pointerf[miny_ind]->y > y )
				miny_ind = i;
			if( pointerf[maxy_ind]->y < y )
				maxy_ind = i;
		}

	if( pointer[0]->x == pointer[total-1]->x &&
		pointer[0]->y == pointer[total-1]->y )
	{
		hullbuf[nout++] = 0;
	}
	else
	{
		// upper half
		int *tl_stack = stack;
		int tl_count = sklansky( pointerf, 0, maxy_ind, tl_stack, -1, 1);
		int *tr_stack = stack + tl_count;
		int tr_count = sklansky( pointerf, total-1, maxy_ind, tr_stack, -1, -1);

		// gather upper part of convex hull to output
		if( !clockwise )
		{
			std::swap( tl_stack, tr_stack );
			std::swap( tl_count, tr_count );
		}

		for( i = 0; i < tl_count-1; i++ )
			hullbuf[nout++] = int(pointer[tl_stack[i]] - data0);
		for( i = tr_count - 1; i > 0; i-- )
			hullbuf[nout++] = int(pointer[tr_stack[i]] - data0);
		int stop_idx = tr_count > 2 ? tr_stack[1] : tl_count > 2 ? tl_stack[tl_count - 2] : -1;

		// lower half
		int *bl_stack = stack;
		int bl_count = sklansky( pointerf, 0, miny_ind, bl_stack, 1, -1);
		int *br_stack = stack + bl_count;
		int br_count = sklansky( pointerf, total-1, miny_ind, br_stack, 1, 1);

		if( clockwise )
		{
			std::swap( bl_stack, br_stack );
			std::swap( bl_count, br_count );
		}

		if( stop_idx >= 0 )
		{
			int check_idx = bl_count > 2 ? bl_stack[1] :
			bl_count + br_count > 2 ? br_stack[2-bl_count] : -1;
			if( check_idx == stop_idx || (check_idx >= 0 &&
											pointer[check_idx]->x == pointer[stop_idx]->x &&
											pointer[check_idx]->y == pointer[stop_idx]->y) )
			{
				// if all the points lie on the same line, then
				// the bottom part of the convex hull is the mirrored top part
				// (except the exteme points).
				bl_count = std::min(bl_count, 2);
				br_count = std::min(br_count, 2);
			}
		}

		for( i = 0; i < bl_count-1; i++ )
			hullbuf[nout++] = int(pointer[bl_stack[i]] - data0);
		for( i = br_count-1; i > 0; i-- )
			hullbuf[nout++] = int(pointer[br_stack[i]] - data0);
	}
	// move result data
	for( i = 0; i < nout; i++ )
		_hull.push_back(data0[hullbuf[i]]);
}

float euclidean_clustering::minAreaRectAngle(std::vector<Point2D>& points)
{
	float angle = 0.0f;
	std::vector<Point2D> hull;
	Point2D out[3];
	convexHull(points, hull, true, true);
	int n = points.size();
	const Point2D* hpoints = hull.data();
	if( n > 2 )
	{
		rotatingCalipers( hpoints, n, (float*)out );
		angle = (float)atan2( (double)out[1].y, (double)out[1].x );
	}
	else if( n == 2 )
	{
		double dx = hpoints[1].x - hpoints[0].x;
		double dy = hpoints[1].y - hpoints[0].y;
		angle = (float)atan2( dy, dx );
	} // angle 0 otherwise
	return (float)(angle*180.0/PI);
}
void euclidean_clustering::initRadiusSearch(const PlainPointCloud& points, bool**  sqr_distances, float radius)
{
	int n = points.size();
	float radius_sqr = radius * radius;
	*sqr_distances = (bool*) malloc(n * n * sizeof(bool));
	for (int j = 0; j < n; j++)
		for (int i = 0; i < n; i++)
		{
			float dx = points[i].x - points[j].x;
			float dy = points[i].y - points[j].y;
			float dz = points[i].z - points[j].z;
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
// 	int nn_start_idx = 0;
// 	int cloudSize = plainPointCloud.size();
//
// 	// indicates the processed status for each point
// 	std::vector<bool> processed (cloudSize, false);
// 	// temporary radius search results
// 	std::vector<int> nn_indices;
// 	// precompute the distance matrix
// 	bool *sqr_distances;
// 	initRadiusSearch(plainPointCloud, &sqr_distances, tolerance);
//
// 	// iterate for all points in the cloud
// 	for (int i = 0; i < cloudSize; ++i)
// 	{
// 		// ignore if already tested
// 		if (processed[i])
// 			continue;
// 		// begin a cluster candidate with one item
// 		std::vector<int> seed_queue;
// 		int sq_idx = 0;
// 		seed_queue.push_back(i);
// 		processed[i] = true;
//
// 		// grow the cluster candidate until all items have been searched through
// 		while (sq_idx < seed_queue.size())
// 		{
// 			int ret = radiusSearch(seed_queue[sq_idx], nn_indices, sqr_distances, cloudSize);
// 			if (!ret)
// 			{
// 				sq_idx++;
// 				continue;
// 			}
// 			// add indices of near points to the cluster candidate
// 			for (size_t j = nn_start_idx; j < nn_indices.size (); ++j)             // can't assume sorted (default isn't!)
// 			{
// 				if (nn_indices[j] == -1 || processed[nn_indices[j]])        // Has this point been processed before ?
// 					continue;
// 				seed_queue.push_back (nn_indices[j]);
// 				processed[nn_indices[j]] = true;
// 			}
// 			sq_idx++;
// 		}
//
// 		// add cluster candidate of fitting size to the resulting clusters
// 		if (seed_queue.size() >= min_pts_per_cluster && seed_queue.size() <= max_pts_per_cluster)
// 		{
// 			PointIndices r;
// 			r.indices.resize(seed_queue.size ());
// 			for (size_t j = 0; j < seed_queue.size (); ++j)
// 				r.indices[j] = seed_queue[j];
// 			std::sort (r.indices.begin (), r.indices.end ());
// 			clusters.push_back(r);
// 		}
// 	}
// 	free(sqr_distances);

	// indicates the processed status for each point
	int pointNo = plainPointCloud.size();
	std::vector<char> processed (pointNo, 0);
	cl::sycl::buffer<Point> cloudBuffer(plainPointCloud.data(), pointNo);
	cl::sycl::buffer<bool, 2> distanceBuffer(cl::sycl::range<2>(pointNo, pointNo));
	cl::sycl::buffer<char> processedBuffer(processed.data(), cl::sycl::range<1>(pointNo));
	cl::sycl::buffer<int> seedQueueBuffer((cl::sycl::range<1>(pointNo)));
	int one = 1;
	cl::sycl::buffer<int> queueLengthBuffer((cl::sycl::range<1>(one)));

	// init distance matrix
	try {
	computeEnv.cmdqueue.submit([&](cl::sycl::handler& h) {
		auto cloud = cloudBuffer.get_access<cl::sycl::access::mode::read>(h);
		auto distances = distanceBuffer.get_access<cl::sycl::access::mode::write>(h);
		float radius = tolerance*tolerance;
		int workGroupNo = pointNo/EPHOS_COMPUTE_GROUP_SIZE + 1;
		//h.parallel_for<euclidean_clustering_init_radius_search>(
			//cl::sycl::range<1>(pointNo),
			//[=](cl::sycl::id<1> item) {
		h.parallel_for<euclidean_clustering_init_radius_search>(
			cl::sycl::nd_range<1>(workGroupNo*EPHOS_COMPUTE_GROUP_SIZE, EPHOS_COMPUTE_GROUP_SIZE),
			[=](cl::sycl::nd_item<1> item) {

			//int iRow = item.get(0);
			int iRow = item.get_global_id(0);
			if (iRow < pointNo) {
				for (int iCol = 0; iCol < pointNo; iCol++) {
					float dx = cloud[iRow].x - cloud[iCol].x;
					float dy = cloud[iRow].y - cloud[iCol].y;
					float dz = cloud[iRow].z - cloud[iCol].z;
					float dist = dx*dx + dy*dy + dz*dz;
					distances[iRow][iCol] = (dist <= radius);
				}
			}
		});
	});
	} catch (cl::sycl::exception& e) {
		std::cerr << e.what() << std::endl;
		exit(-3);
	}
	for (int iStart = 0; iStart < pointNo; iStart++) {
		if (processed[iStart]) {
			continue;
		}
		// start with a new cluster
		processed[iStart] = 1;
		int newQueueLength = 1;
		int oldQueueLength = 0;
		{

			auto queueLengthStorage = queueLengthBuffer.get_access<cl::sycl::access::mode::atomic>();
			cl::sycl::atomic_store(queueLengthStorage[0], 1);
		}
		// grow the cluster
		for (int iQueue = 0; iQueue < newQueueLength; iQueue = oldQueueLength) {
			// radius search
			// mark near points
			int iQueueStart = oldQueueLength;
			int fixedQueueLength = newQueueLength;
			try {
			computeEnv.cmdqueue.submit([&](cl::sycl::handler& h) {
				auto distances = distanceBuffer.get_access<cl::sycl::access::mode::read>(h);
				auto seedQueue = seedQueueBuffer.get_access<cl::sycl::access::mode::read_write>(h);
				auto processed = processedBuffer.get_access<cl::sycl::access::mode::read_write>(h);
				auto nextQueueLength = queueLengthBuffer.get_access<cl::sycl::access::mode::atomic>(h);
				int workGroupNo = pointNo/EPHOS_COMPUTE_GROUP_SIZE + 1;
				//h.parallel_for<euclidean_clustering_radius_search>(
					//cl::sycl::range<1>(pointNo),
					//[=](cl::sycl::id<1> item) {
				h.parallel_for<euclidean_clustering_radius_search>(
					cl::sycl::nd_range<1>(workGroupNo*EPHOS_COMPUTE_GROUP_SIZE, EPHOS_COMPUTE_GROUP_SIZE),
					[=](cl::sycl::nd_item<1> item) {

					//int iCloud = item.get(0);
					int iCloud = item.get_global_id(0);
					if (iCloud < pointNo) {
						if (iCloud == 0 && iQueueStart == 0) {
							// make the device side consistent with the host
							// this step is necessary on the first seed queue grow iteration
							// initialization for first queue element
							processed[iStart] = 1;
							seedQueue[0] = iStart;
						}
						// find out whether a point is a member of the seed queue
						// search through points that are not yet part of a cluster
						// and avoid to add the initial element multiple times
						if (processed[iCloud] == 0 && iCloud != iStart) {
							bool near = false;
							if (iQueueStart == 0) {
								// the first seed queue element might not yet be a member of the seed queue
								// as it is written by only one thread
								// compare distance with the the initial seed queue element
								if (distances[iCloud][iStart]) {
									near = true;
								}
								// no more comparisons necessary as this is the only element
								// in the seed queue in the first iteration
							} else {
								// use the standard search method for subsequent kernel calls
								for (int iQueue = iQueueStart; iQueue < fixedQueueLength && !near; iQueue++) {
									if (distances[iCloud][seedQueue[iQueue]]) {
										near = true;
									}
								}
							}
							if (near) {
								// mark a near point as processed
								// and add it to the seed queue
								processed[iCloud] = 1;
								int iQueue = atomic_fetch_add(nextQueueLength[0], 1);
								seedQueue[iQueue] = iCloud;
							}
						}
					}
				});
			});
			} catch (cl::sycl::exception& e) {
				std::cerr << e.what() << std::endl;
				exit(-3);
			}
			{
				// update the queue length on the host
				oldQueueLength = newQueueLength;

				auto queueLengthStorage = queueLengthBuffer.get_access<cl::sycl::access::mode::atomic>();
				//auto processedStorage = processedBuffer.get_access<cl::sycl::access::mode::read>();
				newQueueLength = cl::sycl::atomic_load(queueLengthStorage[0]);
			}
		}
		// add the seed queue as cluster if it is satisfactory
		if (newQueueLength >= min_pts_per_cluster && newQueueLength <= max_pts_per_cluster) {

			auto seedQueueStorage = seedQueueBuffer.get_access<cl::sycl::access::mode::read>();
			int iCluster = clusters.size();
			clusters.resize(iCluster + 1);
			PointIndices& cluster = clusters[iCluster];
			cluster.indices.resize(newQueueLength);
			for (int i = 0; i < newQueueLength; i++) {
				int iPoint = seedQueueStorage[i];
				cluster.indices[i] = iPoint;
				processed[iPoint] = 1;
			}
		} else {
			auto processedStorage = processedBuffer.get_access<cl::sycl::access::mode::read>();
			for (int i = 0; i < pointNo; i++) {
				if (processedStorage[i] != 0) {
					processed[i] = 1;
				}
			}
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
	if (plainPointCloud.size() == 0)
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
	extract (plainPointCloud,
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
			p.x = (plainPointCloud)[*pit].x;
			p.y = (plainPointCloud)[*pit].y;
			p.z = (plainPointCloud)[*pit].z;
			p.r = 10;
			p.g = 20;
			p.b = 30;
			centroid.x += (plainPointCloud)[*pit].x;
			centroid.y += (plainPointCloud)[*pit].y;
			centroid.z += (plainPointCloud)[*pit].z;

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

void euclidean_clustering::segmentByDistance(
	const PlainPointCloud& plainPointCloud,
	ColorPointCloud& colorPointCloud,
	BoundingboxArray& clusterBoundingBoxes,
	Centroid& clusterCentroids)
{
	PlainPointCloud cloud_segments_array[5];
	double thresholds[5] = {0.5, 1.1, 1.6, 2.3, 2.6f};

	for (const Point& p : plainPointCloud) {

		// categorize by distance from origin
		float origin_distance = p.x*p.x + p.y*p.y;
		if (origin_distance < 15*15 ) {
			cloud_segments_array[0].push_back(p);
		}
		else if(origin_distance < 30*30) {
			cloud_segments_array[1].push_back(p);
		}
		else if(origin_distance < 45*45) {
			cloud_segments_array[2].push_back(p);
		}
		else if(origin_distance < 60*60) {
			cloud_segments_array[3].push_back(p);
		} else {
			cloud_segments_array[4].push_back(p);
		}
	}
	// perform clustering and coloring on the individual categories
	for(unsigned int i=0; i<5; i++)
	{
		clusterAndColor(cloud_segments_array[i], colorPointCloud, clusterBoundingBoxes, clusterCentroids, thresholds[i]);
	}
// 	PlainPointCloud* cloud_segments_array[5];
// 	double thresholds[5] = {0.5, 1.1, 1.6, 2.3, 2.6f};
//
// 	for(unsigned int i=0; i<5; i++)
// 	{
// 		PlainPointCloud *tmp_cloud = new PlainPointCloud();
// 		cloud_segments_array[i] = tmp_cloud;
// 	}
// 	for (unsigned int i = 0; i < plainPointCloud.size(); i++)
// 	{
// 		Point current_point;
// 		current_point.x = plainPointCloud[i].x;
// 		current_point.y = plainPointCloud[i].y;
// 		current_point.z = plainPointCloud[i].z;
//
// 		// categorize by distance from origin
// 		float origin_distance = sqrt(current_point.x*current_point.x + current_point.y*current_point.y);
// 		if (origin_distance < 15 ) {
// 			cloud_segments_array[0]->push_back (current_point);
// 		}
// 		else if(origin_distance < 30) {
// 			cloud_segments_array[1]->push_back (current_point);
// 		}
// 		else if(origin_distance < 45) {
// 			cloud_segments_array[2]->push_back (current_point);
// 		}
// 		else if(origin_distance < 60) {
// 			cloud_segments_array[3]->push_back (current_point);
// 		} else {
// 			cloud_segments_array[4]->push_back (current_point);
// 		}
// 	}
// 	// perform clustering and coloring on the individual categories
// 	for(unsigned int i=0; i<5; i++)
// 	{
// 		clusterAndColor(*cloud_segments_array[i], cloud_segments_array[i]->size(), colorPointCloud, in_clusterBoundingBoxes, in_clusterCentroids, thresholds[i]);
// 	}
}





void euclidean_clustering::init() {
	std::cout << "init\n";
	// try to open input and output file streams
	input_file.exceptions ( std::ifstream::failbit | std::ifstream::badbit );
	output_file.exceptions ( std::ifstream::failbit | std::ifstream::badbit );
	try {
		input_file.open("../../../data/ec_input.dat", std::ios::binary);
	} catch (std::ifstream::failure) {
		std::cerr << "Error opening the input data file" << std::endl;
		exit(-3);
	}
	try {
		output_file.open("../../../data/ec_output.dat", std::ios::binary);
	}  catch (std::ifstream::failure) {
		std::cerr << "Error opening the output data file" << std::endl;
		exit(-3);
	}
#ifdef EPHOS_TESTDATA_GEN
	try {
		datagen_file.open("../../../data/ec_output_gen.dat", std::ios::binary);
	} catch (std::ofstream::failure) {
		std::cerr << "Error opening the generated data file" << std::endl;
		exit(-3);
	}
#endif
	// consume the number of testcases from the input file
	try {
		testcases = read_number_testcases(input_file);
	} catch (std::ios_base::failure& e) {
		std::cerr << e.what() << std::endl;
		exit(-3);
	}
#ifdef EPHOS_TESTCASE_LIMIT
	if (EPHOS_TESTCASE_LIMIT < testcases) {
		testcases = EPHOS_TESTCASE_LIMIT;
	}
#endif // TESTCASE_LIMIT
	// prepare for the first iteration
	error_so_far = false;
	max_delta = 0.0;
	plainPointCloud.clear();
	colorPointCloud.clear();
	clusterBoundingBoxes.clear();
	clusterCentroids.clear();
	// find sycl platform
	try {
		computeEnv = ComputeTools::find_compute_platform(EPHOS_PLATFORM_NAME_S, EPHOS_DEVICE_NAME_S,
			EPHOS_DEVICE_TYPE_S);
		std::cout << "SYCL device name: ";
		std::cout << computeEnv.device.get_info<cl::sycl::info::device::name>() << std::endl;
	} catch (std::logic_error& e) {
		std::cerr << "Failed to initialize SYCL: " << e.what() << std::endl;
	}

	std::cout << "done" << std::endl;
}
void euclidean_clustering::quit() {
	// close data streams
	try {
		input_file.close();
	} catch (std::ifstream::failure& e) {
	}
	try {
		output_file.close();
	} catch (std::ifstream::failure& e) {
	}

}
