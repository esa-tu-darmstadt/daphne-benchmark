#include "benchmark.h"
#include "datatypes.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <omp.h>

// algorithm parameters
const int _cluster_size_min = 20;
const int _cluster_size_max = 100000;
const bool _pose_estimation = true;

// maximum allowed deviation from the reference data
#define MAX_EPS 0.001
#define PARALLEL_REGION_SIZE 40

class euclidean_clustering : public kernel {
private:
	int targetDeviceId = 0;
	int hostDeviceId = 0;
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
	
	void clusterAndColor(const PointCloud *in_cloud_ptr,
		PointCloudRGB *out_cloud_ptr,
		BoundingboxArray *in_out_boundingbox_array,
		Centroid *in_out_centroids,
		double in_max_cluster_distance);
	/**
	* Cluster the point cloud according to the pairwise point distances.
	* Clustering of the same input data is performed multiple times with different thresholds
	* so that points farther away in the cloud also get assigned to a cluster.
	*/
	void segmentByDistance(const PointCloud *in_cloud_ptr,
		PointCloudRGB *out_cloud_ptr,
		BoundingboxArray *in_out_boundingbox_array,
		Centroid *in_out_centroids,
		double in_max_cluster_distance);
		
	void extract(const PointCloud* input_, std::vector<PointIndices> &clusters, double cluster_tolerance_);

	void extractEuclideanClusters (
		const PointCloud &cloud, 
		float tolerance, std::vector<PointIndices> &clusters,
		unsigned int min_pts_per_cluster, 
		unsigned int max_pts_per_cluster);

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

int euclidean_clustering::read_number_testcases(std::ifstream& input_file)
{
	int32_t number;
	try {
		input_file.read((char*)&(number), sizeof(int32_t));
	} catch (std::ifstream::failure) {
		throw std::ios_base::failure("Error reading the number of testcases");
	}
	return number;
}

/**
 * Helper function for calculating the enclosing rectangle with minimum area.
 */
static void rotatingCalipers(const Point2D* points, int n, float* out)
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
	
	// select i suitable points
	for(i = 0; i < n; i++)
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
			// get next base
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
				throw std::logic_error("Error in rotatingCalipers function: main_element should be 0, 1, 2 or 3");
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

/**
 * Helper function for convex hull calculation
 */
static int sklansky(Point2D** array, int start, int end, int* stack, int nsign, int sign2)
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
 * Extension for for point comparison
 */
struct CHullCmpPoints
{
	/**
	 * Compares two points.
	 * Performs a primary test for the x component
	 * and a secondary test for the y component.
	 */
    bool operator()(const Point2D* p1, const Point2D* p2) const
    { 
		return p1->x < p2->x || (p1->x == p2->x && p1->y < p2->y);
	}
};


/**
 * Helper function that computes the convex hull.
 */
void convexHull(
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
	std::sort(pointerf, pointerf + total, CHullCmpPoints());
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

/**
 * Computes the rotation angle of the rectangle with minimum area.
 * return: the rotation angle in degrees
 */
float minAreaRectAngle(std::vector<Point2D>& points)
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

/**
 * Computes the pairwise squared distances. Results are stored in a matrix.
 * An entry of that matrix indicates whether the distance of the two described points 
 * is less or equal to the reference distance.
 * points: points for which we need pairwise distances with size N
 * sqr_distances: resulting distance matrix of size N*N
 * radius: reference distance
 */
void initRadiusSearch(const std::vector<Point> &points, bool**  sqr_distances, const double radius)
{
	int n = points.size();
	*sqr_distances = (bool*) malloc(n * n * sizeof(bool));
	float sqr_radius = radius * radius;
	#pragma omp parallel for default(none) shared(points, sqr_distances, n, sqr_radius) schedule(dynamic)
	for (int j = 0; j < n; j++) {
		for (int i = 0; i < n; i++){
			float dx = points[i].x - points[j].x;
			float dy = points[i].y - points[j].y;
			float dz = points[i].z - points[j].z;
			float sqr_dist = dx*dx + dy*dy + dz*dz;
			(*sqr_distances)[j*n+i] = sqr_dist <= sqr_radius;
		}
		// a point is not near to itself
		(*sqr_distances)[j*n+j] = false;
	}
}



/**
 * Performs radius search for a single point using a precomputed distance matrix.
 * point_index: reference point
 * indices: indices of near points
 * sqr_distances: distance matrix
 * processed: indicates whether a point has been looked at
 * total_points: number of points in the cloud
 * return: the number of near points
 */
int radiusSearch(
	const int point_index, std::vector<int> & indices, const bool* sqr_distances, 
	bool* processed, int total_points)
{
    indices.clear();
    int smaller_index_neighbour = total_points + 1;
    int bigger_index_neighbour = total_points + 1;
	// search through lower indices in the row
    for (int i = 0; i < point_index; i++){
        if (sqr_distances[point_index*total_points +i] && !processed[i]){
                indices.push_back(i);
        }
    }
    // search through higher indices in the row
    for (int i = point_index+1; i < total_points; i++){
        if (sqr_distances[point_index*total_points +i] && !processed[i]){
                indices.push_back(i);
        }
    }
    return indices.size();
}

/**
 * Finds all clusters in the given point cloud that are conformant to the given parameters.
 * cloud: point cloud to cluster
 * tolerance: search radius around a single point
 * clusters: list of resulting clusters
 * min_pts_per_cluster: lower cluster size restriction
 * max_pts_per_cluster: higher cluster size restriction
 */
void euclidean_clustering::extractEuclideanClusters (
	const PointCloud &cloud, 
	float tolerance, std::vector<PointIndices> &clusters,
	unsigned int min_pts_per_cluster, 
	unsigned int max_pts_per_cluster)
{
	// data structures for cluster extraction
	int cloud_size = cloud.size();
	clusters.clear();
	bool* processedStorage = new bool[cloud_size];
	int* clusterCandidateStorage = new int[cloud_size];
	bool* clusterAssignmentStorage = new bool[cloud_size];
	#pragma omp parallel for default(none) shared(cloud_size, processedStorage)
	for (int i = 0; i < cloud_size; ++i){
		processedStorage[i] = false;
	}
	// compute the pairwise distance matrix
	bool* sqrDistanceStorage = new bool[cloud_size*cloud_size];
	const Point* points = cloud.data();
	#pragma omp parallel for default(none) shared(points, sqrDistanceStorage, cloud_size, tolerance)
	for (int j = 0; j < cloud_size; j++) {
		// determine pairwise point distances
		for (int i = 0; i < cloud_size; i++) {
			float dx = points[i].x - points[j].x;
			float dy = points[i].y - points[j].y;
			float dz = points[i].z - points[j].z;
			float dist = dx*dx + dy*dy + dz*dz;
			sqrDistanceStorage[j*cloud_size + i] = dist <= tolerance*tolerance;
		}
		// make a point far away from itself
		sqrDistanceStorage[j*cloud_size + j] = false;
	}
	bool* sqrDistanceBuffer = (bool*)omp_target_alloc(sizeof(bool)*cloud_size*cloud_size, targetDeviceId);
	omp_target_memcpy(sqrDistanceBuffer, sqrDistanceStorage, sizeof(bool)*cloud_size*cloud_size, 
		0, 0, targetDeviceId, hostDeviceId);
	bool* processedBuffer = (bool*)omp_target_alloc(sizeof(bool)*cloud_size, targetDeviceId);
	//bool* processedBuffer = processedStorage;
	omp_target_memcpy(processedBuffer, processedStorage, sizeof(bool)*cloud_size, 
		0, 0, targetDeviceId, hostDeviceId);
	//omp_target_memcpy(clusterAssignmentBuffer, clusterAssignmentStorage, sizeof(int)*cloud_size,
	//	0, 0, targetDeviceId, hostDeviceId);
	//int* clusterAssignmentBuffer = clusterAssignmentStorage;
	int* clusterCandidateBuffer = (int*)omp_target_alloc(sizeof(int)*cloud_size, targetDeviceId);
	//int* clusterCandidateBuffer = new int[cloud_size];
	float sqrTolerance = tolerance*tolerance;
	//const Point* points = cloud.data();
	/*#pragma omp target data \
	map(to: cloud_size, sqrTolerance) \
	map(to: points[:cloud_size])
	{
		// testing with the small set we had better performance without collapse and explicit simd
		#pragma omp target teams distribute parallel for \
		default(none) \
		firstprivate(cloud_size, sqrTolerance) \
		shared(sqrDistanceBuffer, points) \
		is_device_ptr(sqrDistanceBuffer)
		for (int j = 0; j < cloud_size; j++) {
			for (int i = 0; i < cloud_size; i++) {
				float d[3];
				float* p1 = (float*)&points[i];
				float* p2 = (float*)&points[j];
				//#pragma omp simd
				for (int k = 0; k < 3; k++) {
					d[k] = (p1[k] - p2[k]);
				}
				float dist = d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
				sqrDistanceBuffer[j*cloud_size + i] = (dist <= sqrTolerance);
			}
		}
		#pragma omp target teams distribute parallel for \
		default(none) \
		firstprivate(cloud_size) \
		shared(processedBuffer, sqrDistanceBuffer) \
		is_device_ptr(processedBuffer, sqrDistanceBuffer)
		for (int i = 0; i < cloud_size; i++) {
			processedBuffer[i] = false;
			// make a point far away from itself
			//sqrDistanceBuffer[i*cloud_size + i] = false;
		}
	}*/
		
	
	// progress indicators
	// process all points
	for (int i = 0; i < cloud_size; ++i)
	{
		// discard the iteration for points that have already been looked at
		if (processedStorage[i]) {
			continue;
		}
		// begin with a cluster of one element
		int staticCandidateSize = 0;
		int nextCandidateSize = 1;
		int* pNextCandidateSize = &nextCandidateSize;
		int iNextPivot = 0;
		processedStorage[i] = true;
		clusterCandidateStorage[0] = i;
		while (staticCandidateSize < nextCandidateSize) {
			// move new candidate members to buffer
			int deltaCandidateSize = (nextCandidateSize - staticCandidateSize);
			int candidateStart = staticCandidateSize;
			staticCandidateSize = nextCandidateSize;
			
			omp_target_memcpy(clusterCandidateBuffer, clusterCandidateStorage, sizeof(int)*deltaCandidateSize,
				sizeof(int)*candidateStart, sizeof(int)*candidateStart, targetDeviceId, hostDeviceId);
			
			#pragma omp target \
			map(to: candidateStart, staticCandidateSize, cloud_size) \
			map(from: clusterAssignmentStorage[:cloud_size]) \
			is_device_ptr(processedBuffer, clusterCandidateBuffer, sqrDistanceBuffer)
			#pragma omp teams distribute parallel for \
			firstprivate(cloud_size, candidateStart, staticCandidateSize) \
			shared(processedBuffer, clusterCandidateBuffer, sqrDistanceBuffer, clusterAssignmentStorage)
			for (int iCloud = 0; iCloud < cloud_size; iCloud++) {
				if (processedBuffer[iCloud]) {
					continue;
				}
				bool skip = false;
				for (int iCandidate = candidateStart; iCandidate < staticCandidateSize; iCandidate++) {
					if (clusterCandidateBuffer[iCandidate] == iCloud) {
						skip = true;
					}
				}
				
				bool near = false;
				if (!skip) {
					for (int iCandidate = candidateStart; iCandidate < staticCandidateSize; iCandidate++) {
						if (sqrDistanceBuffer[iCloud*cloud_size + clusterCandidateBuffer[iCandidate]]) {
							near = true;
						}
					}
				}
				clusterAssignmentStorage[iCloud] = near;
				if (near) {
					processedBuffer[iCloud] = true;
				}
			}
			for (int iAssign = 0; iAssign < cloud_size; iAssign++) {
				if (clusterAssignmentStorage[iAssign]) {
					if (!processedStorage[iAssign]) {
						clusterCandidateStorage[nextCandidateSize] = iAssign;
						nextCandidateSize += 1;
						processedStorage[iAssign] = true;
					}
				}
			}
		}
		if (staticCandidateSize >= min_pts_per_cluster && staticCandidateSize <= max_pts_per_cluster) {
			int clusterNo = clusters.size();
			clusters.resize(clusterNo + 1);
			PointIndices& cluster = clusters[clusterNo];
			cluster.indices.resize(staticCandidateSize);
			int* clusterData = cluster.indices.data();
			//omp_target_memcpy(candidateStorage, clusterCandidateBuffer, sizeof(int)*staticCandidateSize,
			//	0, 0, hostDeviceId, targetDeviceId);
			//#pragma omp parallel for default(none) shared(processedStorage, candidateStorage, staticCandidateSize)
			//for (int i = 0; i < staticCandidateSize; i++) {
			//	clusterData[i] = clusterCandidateStorage[i];
			//}
			std::memcpy(clusterData, clusterCandidateStorage, sizeof(int)*staticCandidateSize);
			std::sort(cluster.indices.begin(), cluster.indices.end());
		}
	}
	omp_target_free(clusterCandidateBuffer, targetDeviceId);
	omp_target_free(sqrDistanceBuffer, targetDeviceId);
	omp_target_free(processedBuffer, targetDeviceId);
	delete clusterCandidateStorage;
	delete clusterAssignmentStorage;
	delete processedStorage;
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
void euclidean_clustering::extract (const PointCloud *input_, std::vector<PointIndices> &clusters, double cluster_tolerance_)
{
	if (input_->empty())
	{
		clusters.clear ();
		return;
	}
	// Send the input dataset to the spatial locator
	extractEuclideanClusters (*input_, static_cast<float> (cluster_tolerance_), clusters,
		_cluster_size_min, _cluster_size_max );
	// Sort the clusters based on their size (largest one first)
	std::sort (clusters.rbegin (), clusters.rend (), comparePointClusters);
}

/**
 * Performs clustering and coloring on a point cloud
 */
void euclidean_clustering::clusterAndColor(
	const PointCloud *in_cloud_ptr,
	PointCloudRGB *out_cloud_ptr,
	BoundingboxArray* in_out_boundingbox_array,
	Centroid* in_out_centroids,
	double in_max_cluster_distance=0.5)
{
	std::vector<PointIndices> cluster_indices;
	
	// perform expensive radius search
	extract (in_cloud_ptr, cluster_indices, in_max_cluster_distance);

	// color the clusters
	int j = 0;
	unsigned int k = 0;
	for (auto it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
	{
		PointCloudRGB *current_cluster = new PointCloudRGB;//coord + color cluster
		//assign color to each cluster
		PointDouble centroid = {0.0, 0.0, 0.0};
		for (auto pit = it->indices.begin(); pit != it->indices.end(); ++pit)
		{
			// fill new colored cluster point by point
			PointRGB p;
			p.x = (*in_cloud_ptr)[*pit].x;
			p.y = (*in_cloud_ptr)[*pit].y;
			p.z = (*in_cloud_ptr)[*pit].z;
			p.r = 10;
			p.g = 20;
			p.b = 30;

			centroid.x += (*in_cloud_ptr)[*pit].x;
			centroid.y += (*in_cloud_ptr)[*pit].y;
			centroid.z += (*in_cloud_ptr)[*pit].z;

			current_cluster->push_back(p);
		}
		// centroid from mean
		centroid.x /= it->indices.size();
		centroid.y /= it->indices.size();
		centroid.z /= it->indices.size();

		// minimum and maximum extends
		float min_x=std::numeric_limits<float>::max();float max_x=-std::numeric_limits<float>::max();
		float min_y=std::numeric_limits<float>::max();float max_y=-std::numeric_limits<float>::max();
		float min_z=std::numeric_limits<float>::max();float max_z=-std::numeric_limits<float>::max();
		for(unsigned int i=0; i<current_cluster->size();i++)
		{
			if((*current_cluster)[i].x<min_x)  min_x = (*current_cluster)[i].x;
			if((*current_cluster)[i].y<min_y)  min_y = (*current_cluster)[i].y;
			if((*current_cluster)[i].z<min_z)  min_z = (*current_cluster)[i].z;
			if((*current_cluster)[i].x>max_x)  max_x = (*current_cluster)[i].x;
			if((*current_cluster)[i].y>max_y)  max_y = (*current_cluster)[i].y;
			if((*current_cluster)[i].z>max_z)  max_z = (*current_cluster)[i].z;
		}
		float l = max_x - min_x;
		float w = max_y - min_y;
		float h = max_z - min_z;
		
		// create bounding box from cluster extends
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
			for (unsigned int i=0; i<current_cluster->size(); i++)
			{
				Point2D ip;
				ip.x = ((*current_cluster)[i].x + fabs(min_x))*8;
				ip.y = ((*current_cluster)[i].y + fabs(min_y))*8;
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
		bounding_box.orientation.x = 0.0; //x
		bounding_box.orientation.y = 0.0; //y
		bounding_box.orientation.z = sinYaw; //z
		bounding_box.orientation.w = cosYaw; //w, formerly yzx

		if (  bounding_box.dimensions.x >0 && bounding_box.dimensions.y >0 && bounding_box.dimensions.z > 0 &&
			bounding_box.dimensions.x < 15 && bounding_box.dimensions.y >0 && bounding_box.dimensions.y < 15 &&
			max_z > -1.5 && min_z > -1.5 && min_z < 1.0 )
		{
			in_out_boundingbox_array->boxes.push_back(bounding_box);
			in_out_centroids->points.push_back(centroid);
		}
		out_cloud_ptr->insert(out_cloud_ptr->end(), current_cluster->begin(), current_cluster->end());
		j++; k++;
	}
}
/**
 * Segments the cloud into categories representing distance ranges from the origin
 * and performs clustering and coloring on the individual categories.
 * in_cloud_ptr: point cloud
 * out_cloud_ptr: resulting point cloud
 * out_boundingbox_array: resulting bounding boxes
 * in_out_centroids: resulting cluster centroids
 * in_max_cluster_distance: distance threshold
 */
void euclidean_clustering::segmentByDistance(
	const PointCloud *in_cloud_ptr,
	PointCloudRGB *out_cloud_ptr,
	BoundingboxArray *out_boundingbox_array,
	Centroid *in_out_centroids,
	double in_max_cluster_distance=0.5)
{
	PointCloud*   cloud_segments_array[5];
	double thresholds[5] = {0.5, 1.1, 1.6, 2.3, 2.6f};

	for(unsigned int i=0; i<5; i++)
	{
		PointCloud *tmp_cloud = new PointCloud;
		cloud_segments_array[i] = tmp_cloud;
	}
	for (unsigned int i=0; i<in_cloud_ptr->size(); i++)
	{
		Point current_point;
		current_point.x = (*in_cloud_ptr)[i].x;
		current_point.y = (*in_cloud_ptr)[i].y;
		current_point.z = (*in_cloud_ptr)[i].z;
		
		// categorize by distance from origin
		float origin_distance = sqrt(current_point.x*current_point.x + current_point.y*current_point.y);
		if (origin_distance < 15 ) { 
			cloud_segments_array[0]->push_back (current_point);
		}
		else if(origin_distance < 30) {
			cloud_segments_array[1]->push_back (current_point);
		}
		else if(origin_distance < 45) {
			cloud_segments_array[2]->push_back (current_point);
		}
		else if(origin_distance < 60) {
			cloud_segments_array[3]->push_back (current_point);
		} else {
			cloud_segments_array[4]->push_back (current_point);
		}
	}
	// perform clustering and coloring on the individual categories
	for(unsigned int i=0; i<5; i++)
	{
		clusterAndColor(cloud_segments_array[i], out_cloud_ptr, out_boundingbox_array, in_out_centroids, thresholds[i]);
	}
}

/**
 * Reads the next point cloud.
 */
void parsePointCloud(std::ifstream& input_file, PointCloud *cloud)
{
	int size = 0;
	Point p;
	input_file.read((char*)&(size), sizeof(int));
	try {
		for (int i = 0; i < size; i++)
		{
			input_file.read((char*)&p.x, sizeof(float));
			input_file.read((char*)&p.y, sizeof(float));
			input_file.read((char*)&p.z, sizeof(float));
			cloud->push_back(p);
		}
	} catch (std::ifstream::failure) {
		throw std::ios_base::failure("Error reading point cloud");
	}
}

/**
 * Reads the next reference cloud result.
 */
void parseOutCloud(std::ifstream& input_file, PointCloudRGB *cloud)
{
    int size = 0;
    PointRGB p;
    try {
	input_file.read((char*)&(size), sizeof(int));

	for (int i = 0; i < size; i++)
	    {
		input_file.read((char*)&p.x, sizeof(float));
		input_file.read((char*)&p.y, sizeof(float));
		input_file.read((char*)&p.z, sizeof(float));
		input_file.read((char*)&p.r, sizeof(uint8_t));
		input_file.read((char*)&p.g, sizeof(uint8_t));
		input_file.read((char*)&p.b, sizeof(uint8_t));				    
		cloud->push_back(p);
	    }
    }  catch (std::ifstream::failure) {
		throw std::ios_base::failure("Error reading reference cloud");
    } 
}

/**
 * Reads the next reference bounding boxes.
 */
void parseBoundingboxArray(std::ifstream& input_file, BoundingboxArray *bb_array)
{
    int size = 0;
    Boundingbox bba;
    try {
	input_file.read((char*)&(size), sizeof(int));

	for (int i = 0; i < size; i++)
	    {
		input_file.read((char*)&bba.position.x, sizeof(double));
		input_file.read((char*)&bba.position.y, sizeof(double));
		input_file.read((char*)&bba.orientation.x, sizeof(double));
		input_file.read((char*)&bba.orientation.y, sizeof(double));
		input_file.read((char*)&bba.orientation.z, sizeof(double));
		input_file.read((char*)&bba.orientation.w, sizeof(double));
		input_file.read((char*)&bba.dimensions.x, sizeof(double));
		input_file.read((char*)&bba.dimensions.y, sizeof(double));
		bb_array->boxes.push_back(bba);
	    }
    }  catch (std::ifstream::failure) {
		throw std::ios_base::failure("Error reading reference bounding boxes");
    }
}

/*
 * Reads the next reference centroids.
 */
void parseCentroids(std::ifstream& input_file, Centroid *centroids)
{
	int size = 0;
	PointDouble p;
	try {
		input_file.read((char*)&(size), sizeof(int));
		for (int i = 0; i < size; i++)
		{
			input_file.read((char*)&p.x, sizeof(double));
			input_file.read((char*)&p.y, sizeof(double));
			input_file.read((char*)&p.z, sizeof(double));
			centroids->points.push_back(p);
		}
	} catch (std::ifstream::failure) {
		throw std::ios_base::failure("Error reading reference centroids");
	}
}

int euclidean_clustering::read_next_testcases(int count)
{
	int i;
	// free previously allocated memory
	delete [] in_cloud_ptr;
	delete [] out_cloud_ptr;
	delete [] out_boundingbox_array;
	delete [] out_centroids;
	// allocate new memory for the current case
	in_cloud_ptr = new PointCloud[count];
	out_cloud_ptr = new PointCloudRGB[count];
	out_boundingbox_array = new BoundingboxArray[count];
	out_centroids = new Centroid[count];
	// read the testcase data
	for (i = 0; (i < count) && (read_testcases < testcases); i++,read_testcases++)
	{
		try {
			parsePointCloud(input_file, in_cloud_ptr + i);
		} catch (std::ios_base::failure& e) {
			std::cerr << e.what() << std::endl;
			exit(-3);
		}
	}
	return i;
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
	// consume the number of testcases from the input file
	try {
		testcases = read_number_testcases(input_file);
	} catch (std::ios_base::failure& e) {
		std::cerr << e.what() << std::endl;
		exit(-3);
	}
	// select device
	#ifdef EPHOS_HOST_DEVICE_ID
		hostDeviceId = EPHOS_HOST_DEVICE_ID;
	#else
		hostDeviceId = omp_get_initial_device();
	#endif
	#ifdef EPHOS_TARGET_DEVICE_ID
		targetDeviceId = EPHOS_TARGET_DEVICE_ID;
	#else
		targetDeviceId = omp_get_default_device();
	#endif
	std::cout << "Selected host device id: " << hostDeviceId << std::endl;
	std::cout << "Selected target device id: " << targetDeviceId << std::endl;
	// prepare for the first iteration
	error_so_far = false;
	max_delta = 0.0;
	in_cloud_ptr = nullptr;
	out_cloud_ptr = nullptr;
	out_boundingbox_array = nullptr;
	out_centroids = nullptr;

	std::cout << "done\n" << std::endl;
}

void euclidean_clustering::run(int p) {
  pause_func();
  
  while (read_testcases < testcases)
    {
      int count = read_next_testcases(p);
      unpause_func();
      for (int i = 0; i < count; i++)
	  {
	      // actual kernel invocation
	      segmentByDistance(&in_cloud_ptr[i],
				&out_cloud_ptr[i],
				&out_boundingbox_array[i],
				&out_centroids[i]);
	  }
      pause_func();
      check_next_outputs(count);
    }
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

void euclidean_clustering::check_next_outputs(int count)
{
	PointCloudRGB reference_out_cloud;
	BoundingboxArray reference_bb_array;
	Centroid reference_centroids;
	
	for (int i = 0; i < count; i++)
	{
		// read the reference result
		try {
			parseOutCloud(output_file, &reference_out_cloud);
			parseBoundingboxArray(output_file, &reference_bb_array);
			parseCentroids(output_file, &reference_centroids);
		} catch (std::ios_base::failure& e) {
			std::cerr << e.what() << std::endl;
			exit(-3);
		}

		// as the result is still right when points/boxes/centroids are in different order,
		// we sort the result and reference to normalize it and we can compare it
		std::sort(reference_out_cloud.begin(), reference_out_cloud.end(), compareRGBPoints);
		std::sort(out_cloud_ptr[i].begin(), out_cloud_ptr[i].end(), compareRGBPoints);
		std::sort(reference_bb_array.boxes.begin(), reference_bb_array.boxes.end(), compareBBs);
		std::sort(out_boundingbox_array[i].boxes.begin(), out_boundingbox_array[i].boxes.end(), compareBBs);
		std::sort(reference_centroids.points.begin(), reference_centroids.points.end(), comparePoints);
		std::sort(out_centroids[i].points.begin(), out_centroids[i].points.end(), comparePoints);
		// test for size differences
		if (reference_out_cloud.size() != out_cloud_ptr[i].size())
		{
			std::cout << "Deviating size (" << out_cloud_ptr[i].size() << "!=";
			std::cout <<  reference_out_cloud.size() << ")" << std::endl;
			error_so_far = true;
			continue;
		}
		if (reference_bb_array.boxes.size() != out_boundingbox_array[i].boxes.size())
		{
			error_so_far = true;
			continue;
		}
		if (reference_centroids.points.size() != out_centroids[i].points.size())
		{
			error_so_far = true;
			continue;
		}
		// test for content divergence
		for (int j = 0; j < reference_out_cloud.size(); j++)
		{
			max_delta = std::fmax(std::abs(out_cloud_ptr[i][j].x - reference_out_cloud[j].x), max_delta);
			max_delta = std::fmax(std::abs(out_cloud_ptr[i][j].y - reference_out_cloud[j].y), max_delta);
			max_delta = std::fmax(std::abs(out_cloud_ptr[i][j].z - reference_out_cloud[j].z), max_delta);
		}
		for (int j = 0; j < reference_bb_array.boxes.size(); j++)
		{
			max_delta = std::fmax(std::abs(out_boundingbox_array[i].boxes[j].position.x - reference_bb_array.boxes[j].position.x), max_delta);		    
			max_delta = std::fmax(std::abs(out_boundingbox_array[i].boxes[j].position.y - reference_bb_array.boxes[j].position.y), max_delta);
			max_delta = std::fmax(std::abs(out_boundingbox_array[i].boxes[j].dimensions.x - reference_bb_array.boxes[j].dimensions.x), max_delta);		    
			max_delta = std::fmax(std::abs(out_boundingbox_array[i].boxes[j].dimensions.y - reference_bb_array.boxes[j].dimensions.y), max_delta); 
			max_delta = std::fmax(std::abs(out_boundingbox_array[i].boxes[j].orientation.x - reference_bb_array.boxes[j].orientation.x), max_delta);
			max_delta = std::fmax(std::abs(out_boundingbox_array[i].boxes[j].orientation.y - reference_bb_array.boxes[j].orientation.y), max_delta);			
		}
		for (int j = 0; j < reference_centroids.points.size(); j++)
		{
			max_delta = std::fmax(std::abs(out_centroids[i].points[j].x - reference_centroids.points[j].x), max_delta);
			max_delta = std::fmax(std::abs(out_centroids[i].points[j].y - reference_centroids.points[j].y), max_delta);
			max_delta = std::fmax(std::abs(out_centroids[i].points[j].z - reference_centroids.points[j].z), max_delta);
		}
		// finishing steps for the next iteration
		reference_bb_array.boxes.clear();
		reference_out_cloud.clear();
		reference_centroids.points.clear();
	}
}

bool euclidean_clustering::check_output() 
{
	std::cout << "checking output \n";

	// acts as complement to init()
	input_file.close();
	output_file.close();
	std::cout << "max delta: " << max_delta << "\n";
	if ((max_delta > MAX_EPS) || error_so_far)
	{
		return false;
	} else 
	{
		return true;
	}
}

// set kernel used by main()
euclidean_clustering a = euclidean_clustering();
kernel& myKernel = a;

