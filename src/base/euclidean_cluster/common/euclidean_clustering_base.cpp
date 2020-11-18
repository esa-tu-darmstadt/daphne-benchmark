
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

#include "euclidean_clustering_base.h"

euclidean_clustering_base::euclidean_clustering_base() :
	benchmark(),
	plainPointCloud(),
	colorPointCloud(),
	clusterBoundingBoxes(),
	clusterCentroids(),
	read_testcases(0),
	input_file(),
	output_file(),
	datagen_file(),
	error_so_far(false),
	max_delta(0)
{}

euclidean_clustering_base::~euclidean_clustering_base() {}


void euclidean_clustering_base::init() {
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
		testcases = read_testdata_signature(input_file, output_file);
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
}
void euclidean_clustering_base::quit() {
	// close data streams
	try {
		input_file.close();
	} catch (std::ifstream::failure& e) {
	}
	try {
		output_file.close();
	} catch (std::ifstream::failure& e) {
	}
#ifdef EPHOS_TESTDATA_GEN
	try {
		datagen_file.close();
	} catch (std::ofstream::failure& e) {
	}
#endif
}


void euclidean_clustering_base::rotatingCalipers( const Point2D* points, int n, float* out )
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

int euclidean_clustering_base::sklansky(
	Point2D** array, int start, int end, int* stack, int nsign, int sign2)
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

void euclidean_clustering_base::convexHull(
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

float euclidean_clustering_base::minAreaRectAngle(std::vector<Point2D>& points)
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

void euclidean_clustering_base::segmentByDistance(
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
	// perform clustering and coloring on the individual segments
	double thresholds[5] = { 0.5, 1.1, 1.6, 2.3, 2.6 };
	for(unsigned int i=0; i<5; i++)
	{
		clusterAndColor(cloudSegments[i], colorPointCloud,
			clusterBoundingBoxes, clusterCentroids, thresholds[i]);
	}
	delete[] cloudSegmentStorage;
}


void euclidean_clustering_base::parsePlainPointCloud(std::ifstream& input_file, PlainPointCloud& cloud)
{
	try {
		int cloudSize = 0;
		input_file.read((char*)&cloudSize, sizeof(int));
		// TODO deallocate
		cloud.data = new Point[cloudSize];
		cloud.capacity = cloudSize;
		cloud.size = cloudSize;

		for (int i = 0; i < cloudSize; i++)
		{
			Point point;
			input_file.read((char*)&(point.x), sizeof(float));
			input_file.read((char*)&(point.y), sizeof(float));
			input_file.read((char*)&(point.z), sizeof(float));
			cloud.data[i] = point;
		}
	} catch (std::ifstream::failure& e) {
		throw std::ios_base::failure("Error reading point cloud");
	}
}
/**
 * Reads the next reference cloud result.
 */
void euclidean_clustering_base::parseColorPointCloud(std::ifstream& input_file, ColorPointCloud& cloud)
{
    try {
		int cloudSize = 0;
		input_file.read((char*)&cloudSize, sizeof(int));
		cloud.data = new PointRGB[cloudSize];
		cloud.capacity = cloudSize;
		cloud.size = cloudSize;

		for (int i = 0; i < cloudSize; i++)
	    {
			PointRGB p;
			input_file.read((char*)&p.x, sizeof(float));
			input_file.read((char*)&p.y, sizeof(float));
			input_file.read((char*)&p.z, sizeof(float));
			input_file.read((char*)&p.r, sizeof(uint8_t));
			input_file.read((char*)&p.g, sizeof(uint8_t));
			input_file.read((char*)&p.b, sizeof(uint8_t));
			cloud.data[i] = p;
	    }
    }  catch (std::ifstream::failure&) {
		throw std::ios_base::failure("Error reading reference cloud");
    }
}


void euclidean_clustering_base::parseBoundingboxArray(std::ifstream& input_file, BoundingboxArray& bb_array)
{
	try {
		int boxNo = 0;
		input_file.read((char*)&boxNo, sizeof(int));
		for (int i = 0; i < boxNo; i++)
		{
			Boundingbox box;
			input_file.read((char*)&box.position.x, sizeof(double));
			input_file.read((char*)&box.position.y, sizeof(double));
			input_file.read((char*)&box.orientation.x, sizeof(double));
			input_file.read((char*)&box.orientation.y, sizeof(double));
			input_file.read((char*)&box.orientation.z, sizeof(double));
			input_file.read((char*)&box.orientation.w, sizeof(double));
			input_file.read((char*)&box.dimensions.x, sizeof(double));
			input_file.read((char*)&box.dimensions.y, sizeof(double));
			bb_array.boxes.push_back(box);
		}
	}  catch (std::ifstream::failure& e) {
		throw std::ios_base::failure("Error reading reference bounding boxes");
	}
}

/*
 * Reads the next reference centroids.
 */
void euclidean_clustering_base::parseCentroids(std::ifstream& input_file, Centroid& centroids)
{
	try {
		int centroidNo = 0;
		input_file.read((char*)&centroidNo, sizeof(int));
		for (int i = 0; i < centroidNo; i++)
		{
			PointDouble p;
			input_file.read((char*)&p.x, sizeof(double));
			input_file.read((char*)&p.y, sizeof(double));
			input_file.read((char*)&p.z, sizeof(double));
			centroids.points.push_back(p);
		}
    } catch (std::ifstream::failure& e) {
		throw std::ios_base::failure("Error reading reference centroids");
    }
}

void euclidean_clustering_base::writeColorPointCloud(std::ofstream& datagen_file, ColorPointCloud& cloud) {
	try {
		int pointNo = cloud.size;
		datagen_file.write((char*)&pointNo, sizeof(int));
		for (int i = 0; i < pointNo; i++) {
			PointRGB p = cloud.data[i];
			datagen_file.write((char*)&p.x, sizeof(float));
			datagen_file.write((char*)&p.y, sizeof(float));
			datagen_file.write((char*)&p.z, sizeof(float));
			datagen_file.write((char*)&p.r, sizeof(uint8_t));
			datagen_file.write((char*)&p.g, sizeof(uint8_t));
			datagen_file.write((char*)&p.b, sizeof(uint8_t));
		}
	} catch (std::ofstream::failure& e) {
		throw std::ios_base::failure("Error writing reference cloud");
	}
}

void euclidean_clustering_base::writeBoundingboxArray(std::ofstream& datagen_file, BoundingboxArray& bb_array) {
	try {
		int boundingBoxNo = bb_array.boxes.size();
		datagen_file.write((char*)&boundingBoxNo, sizeof(int));
		for (Boundingbox b : bb_array.boxes) {
			datagen_file.write((char*)&b.position.x, sizeof(double));
			datagen_file.write((char*)&b.position.y, sizeof(double));
			datagen_file.write((char*)&b.orientation.x, sizeof(double));
			datagen_file.write((char*)&b.orientation.y, sizeof(double));
			datagen_file.write((char*)&b.orientation.z, sizeof(double));
			datagen_file.write((char*)&b.orientation.w, sizeof(double));
			datagen_file.write((char*)&b.dimensions.x, sizeof(double));
			datagen_file.write((char*)&b.dimensions.y, sizeof(double));
		}
	} catch (std::ofstream::failure& e) {
		throw std::ios_base::failure("Error writing reference bounding boxes");
	}

}

void euclidean_clustering_base::writeCentroids(std::ofstream& datagen_file, Centroid& centroids) {
	try {
		int centroidNo = centroids.points.size();
		datagen_file.write((char*)&centroidNo, sizeof(int));
		for (PointDouble p : centroids.points) {
			datagen_file.write((char*)&p.x, sizeof(double));
			datagen_file.write((char*)&p.y, sizeof(double));
			datagen_file.write((char*)&p.z, sizeof(double));
		}
	} catch (std::ofstream::failure& e) {
		throw std::ios_base::failure("Error writing reference centroids");
	}

}

#ifdef EPHOS_TESTDATA_LEGACY
int euclidean_clustering_base::read_testdata_signature(std::ifstream& input_file, std::ifstream& output_file)
{
	int number;
	try {
		input_file.read((char*)&number, sizeof(int32_t));
	} catch (std::ifstream::failure&) {
		throw std::ios_base::failure("Error reading the input data signature");
	}
	return number;
}
#else // EPHOS_TESTDATA_LEGACY
int euclidean_clustering_base::read_testdata_signature(std::ifstream& input_file, std::ifstream& output_file)
{
	int32_t number1, number2, zero, version1, version2;
	try {
		input_file.read((char*)&zero, sizeof(int32_t));
		input_file.read((char*)&version1, sizeof(int32_t));
		input_file.read((char*)&number1, sizeof(int32_t));
	} catch (std::ifstream::failure&) {
		throw std::ios_base::failure("Error reading the input data signature");
	}
	if (zero != 0x0) {
		throw std::ios_base::failure(
			"Misformatted input test data signature. You may be using legacy test data");
	}
	if (version1 != 0x1) {
		throw std::ios_base::failure(
			std::string(
				"Misformatted input test data signature. "
				"Expected test data version 1. "
				"Instead got version ") + std::to_string(version1));
	}
	if (number1 < 0 || number1 > 10000) {
		throw std::ios_base::failure(
			std::string("Unreasonable number of test cases (") +
			std::to_string(number1) +
			std::string(") in input test data"));
	}
	try {
		output_file.read((char*)&zero, sizeof(int32_t));
		output_file.read((char*)&version2, sizeof(int32_t));
		output_file.read((char*)&number2, sizeof(int32_t));
	} catch (std::ifstream::failure) {
		throw std::ios_base::failure("Error reading the output test data signature");
	}
	if (zero != 0x0) {
		throw std::ios_base::failure(
			"Misformatted output test data signature. You may be using legacy test data");
	}
	if (version2 != 0x1) {
		throw std::ios_base::failure(
			std::string(
				"Misformatted output test data signature. "
				"Expected test data version 1. "
				"Instead got version ") +
			std::to_string(version2));
	}
	if (number2 != number1) {
		throw std::ios_base::failure(
			std::string("Number of test cases in output test data (") +
			std::to_string(number2) +
			std::string(") does not match number of test cases input test data (") +
			std::to_string(number1) + std::string(")"));
	}
	return number1;
}
#endif // !EPHOS_TESTDATA_LEGACY

int euclidean_clustering_base::read_next_testcases(int count)
{
	// free memory of the last iteration and allocate new one
	int i;
	plainPointCloud.resize(count);
	colorPointCloud.resize(count);
	clusterBoundingBoxes.resize(count);
	clusterCentroids.resize(count);
	//plainCloudSize.resize(count);

	// read the respective point clouds
	for (i = 0; (i < count) && (read_testcases < testcases); i++,read_testcases++)
	{
		try {
			parsePlainPointCloud(input_file, plainPointCloud[i]);//, plainCloudSize[i]);
		} catch (std::ios_base::failure& e) {
			std::cerr << e.what() << std::endl;
			exit(-3);
		}
	}
	return i;
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

void euclidean_clustering_base::check_next_outputs(int count)
{
	ColorPointCloud refPointCloud;
	BoundingboxArray refBoundingBoxes;
	Centroid refClusterCentroids;

	for (int i = 0; i < count; i++)
	{
		// read the reference result
		try {
			parseColorPointCloud(output_file, refPointCloud);
			parseBoundingboxArray(output_file, refBoundingBoxes);
			parseCentroids(output_file, refClusterCentroids);
#ifdef EPHOS_TESTDATA_GEN
			writeColorPointCloud(datagen_file, colorPointCloud[i]);
			writeBoundingboxArray(datagen_file, clusterBoundingBoxes[i]);
			writeCentroids(datagen_file, clusterCentroids[i]);
#endif
		} catch (std::ios_base::failure& e) {
			std::cerr << e.what() << std::endl;
			exit(-3);
		}

		// as the result is still right when points/boxes/centroids are in different order,
		// we sort the result and reference to normalize it and we can compare it
		//std::sort(refPointCloud.begin(), refPointCloud.end(), compareRGBPoints);
		//std::sort(colorPointCloud[i].begin(), colorPointCloud[i].end(), compareRGBPoints);
		std::sort(refPointCloud.data, refPointCloud.data + refPointCloud.size, compareRGBPoints);
		std::sort(colorPointCloud[i].data, colorPointCloud[i].data + colorPointCloud[i].size, compareRGBPoints);
		std::sort(refBoundingBoxes.boxes.begin(), refBoundingBoxes.boxes.end(), compareBBs);
		std::sort(clusterBoundingBoxes[i].boxes.begin(), clusterBoundingBoxes[i].boxes.end(), compareBBs);
		std::sort(refClusterCentroids.points.begin(), refClusterCentroids.points.end(), comparePoints);
		std::sort(clusterCentroids[i].points.begin(), clusterCentroids[i].points.end(), comparePoints);
		// test for size differences
		std::ostringstream sError;
		int caseErrorNo = 0;
		// test for size differences
		if (refPointCloud.size != colorPointCloud[i].size)
		{
			error_so_far = true;
			caseErrorNo += 1;
			sError << " invalid point number: " << colorPointCloud[i].size;
			sError << " should be " << refPointCloud.size << std::endl;
		}
		if (refBoundingBoxes.boxes.size() != clusterBoundingBoxes[i].boxes.size())
		{
			error_so_far = true;
			caseErrorNo += 1;
			sError << " invalid bounding box number: " << clusterBoundingBoxes[i].boxes.size();
			sError << " should be " << refBoundingBoxes.boxes.size() << std::endl;
		}
		if (refClusterCentroids.points.size() != clusterCentroids[i].points.size())
		{
			error_so_far = true;
			caseErrorNo += 1;
			sError << " invalid centroid number: " << clusterCentroids[i].points.size();
			sError << " should be " << refClusterCentroids.points.size() << std::endl;
		}
		if (caseErrorNo == 0) {
			// test for content divergence
			for (int j = 0; j < refPointCloud.size; j++)
			{
				float deltaX = std::abs(colorPointCloud[i].data[j].x - refPointCloud.data[j].x);
				float deltaY = std::abs(colorPointCloud[i].data[j].y - refPointCloud.data[j].y);
				float deltaZ = std::abs(colorPointCloud[i].data[j].z - refPointCloud.data[j].z);
				float delta = std::fmax(deltaX, std::fmax(deltaY, deltaZ));
				if (delta > EPHOS_MAX_EPS) {
					caseErrorNo += 1;
					sError << " deviating point " << j << ": (";
					sError << colorPointCloud[i].data[j].x << " " << colorPointCloud[i].data[j].y;
					sError << " " << colorPointCloud[i].data[j].z << ") should be (";
					sError << refPointCloud.data[j].x << " " << refPointCloud.data[j].y << " ";
					sError << refPointCloud.data[j].z << ")" << std::endl;
					if (delta > max_delta) {
						max_delta = delta;
					}
				}
			}
			for (int j = 0; j < refBoundingBoxes.boxes.size(); j++)
			{
				float deltaX = std::abs(clusterBoundingBoxes[i].boxes[j].position.x - refBoundingBoxes.boxes[j].position.x);
				float deltaY = std::abs(clusterBoundingBoxes[i].boxes[j].position.y - refBoundingBoxes.boxes[j].position.y);
				float deltaW = std::abs(clusterBoundingBoxes[i].boxes[j].dimensions.x - refBoundingBoxes.boxes[j].dimensions.x);
				float deltaH = std::abs(clusterBoundingBoxes[i].boxes[j].dimensions.y - refBoundingBoxes.boxes[j].dimensions.y);
				float deltaOX = std::abs(clusterBoundingBoxes[i].boxes[j].orientation.x - refBoundingBoxes.boxes[j].orientation.x);
				float deltaOY = std::abs(clusterBoundingBoxes[i].boxes[j].orientation.y - refBoundingBoxes.boxes[j].orientation.y);
				float deltaP = std::fmax(deltaX, deltaY);
				float deltaS = std::fmax(deltaW, deltaH);
				float deltaO = std::fmax(deltaOX, deltaOY);
				float delta = 0;
				if (deltaP > EPHOS_MAX_EPS) {
					delta = std::fmax(delta, deltaP);
					sError << " deviating bounding box " << j << " position: (";
					sError << clusterBoundingBoxes[i].boxes[j].position.x << " ";
					sError << clusterBoundingBoxes[i].boxes[j].position.y << ") should be (";
					sError << refBoundingBoxes.boxes[j].position.x << " ";
					sError << refBoundingBoxes.boxes[j].position.y << ")" << std::endl;
				}
				if (deltaS > EPHOS_MAX_EPS) {
					delta = std::fmax(delta, deltaS);
					sError << " deviating bounding box " << j << " size: (";
					sError << clusterBoundingBoxes[i].boxes[j].dimensions.x << " ";
					sError << clusterBoundingBoxes[i].boxes[j].dimensions.y << ") should be (";
					sError << refBoundingBoxes.boxes[j].dimensions.x << " ";
					sError << refBoundingBoxes.boxes[j].dimensions.y << ")" << std::endl;
				}
				if (deltaO > EPHOS_MAX_EPS) {
					delta = std::fmax(delta, deltaO);
					sError << " deviating bound box " << j << " orientation: (";
					sError << clusterBoundingBoxes[i].boxes[j].orientation.x << " ";
					sError << clusterBoundingBoxes[i].boxes[j].orientation.y << ") should be (";
					sError << refBoundingBoxes.boxes[j].orientation.x << " ";
					sError << refBoundingBoxes.boxes[j].orientation.y << ")" << std::endl;
				}
				if (delta > EPHOS_MAX_EPS) {
					caseErrorNo += 1;
					if (delta > max_delta) {
						max_delta = delta;
					}
				}
			}
			for (int j = 0; j < refClusterCentroids.points.size(); j++)
			{
				float deltaX = std::abs(clusterCentroids[i].points[j].x - refClusterCentroids.points[j].x);
				float deltaY = std::abs(clusterCentroids[i].points[j].y - refClusterCentroids.points[j].y);
				float deltaZ = std::abs(clusterCentroids[i].points[j].z - refClusterCentroids.points[j].z);
				float delta = std::fmax(deltaX, std::fmax(deltaY, deltaZ));
				if (delta > EPHOS_MAX_EPS) {
					caseErrorNo += 1;
					if (delta > max_delta) {
						max_delta = delta;
					}
					sError << " deviating centroid " << j << " position: (";
					sError << clusterCentroids[i].points[j].x << " " << clusterCentroids[i].points[j].y << " ";
					sError << clusterCentroids[i].points[j].z << ") should be (";
					sError << refClusterCentroids.points[j].x << " " << refClusterCentroids.points[j].y << " ";
					sError << refClusterCentroids.points[j].z << ")" << std::endl;
				}
			}
		}
		if (caseErrorNo > 0) {
			std::cerr << "Errors for test case " << read_testcases - count + i;
			std::cerr << " (" << caseErrorNo << "):" << std::endl;
			std::cerr << sError.str() << std::endl;
		}
		// finishing steps for the next iteration
		refBoundingBoxes.boxes.clear();
		refClusterCentroids.points.clear();
		delete[] refPointCloud.data;
		delete[] plainPointCloud[i].data;
		delete[] colorPointCloud[i].data;
	}
	plainPointCloud.clear();
	colorPointCloud.clear();
	clusterBoundingBoxes.clear();
	clusterCentroids.clear();
}

void euclidean_clustering_base::run(int p) {
	std::cout << "executing for " << testcases << " test cases" << std::endl;
	start_timer();
	pause_timer();

	while (read_testcases < testcases)
	{
		// read the next input data
		int count = read_next_testcases(p);
		resume_timer();
		for (int i = 0; i < count; i++)
		{
			// actual kernel invocation
			segmentByDistance(
				plainPointCloud[i],
				//plainCloudSize[i],
				colorPointCloud[i],
				clusterBoundingBoxes[i],
				clusterCentroids[i]
			);
		}
		// pause the timer, then read and compare with the reference data
		pause_timer();
		check_next_outputs(count);
	}
	stop_timer();
}

bool euclidean_clustering_base::check_output()
{
	std::cout << "checking output \n";

	// acts as complement to init()

	std::cout << "max delta: " << max_delta << "\n";
	if ((max_delta > EPHOS_MAX_EPS) || error_so_far)
	{
		return false;
	} else
	{
		return true;
	}
}
