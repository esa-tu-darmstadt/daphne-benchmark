/**
 * Author:  Leonardo Solis, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2018
 * License: Apache 2.0 (see attachached File)
 */

#ifndef EPHOS_DISTANCE_PACKETS_PER_ITEM
#define EPHOS_DISTANCE_PACKETS_PER_ITEM 1
#endif

#ifndef EPHOS_DISTANCES_PER_PACKET
#define EPHOS_DISTANCES_PER_PACKET 1
#endif

#ifdef EPHOS_LINE_PROCESSING
#undef EPHOS_DISTANCE_PACKETS_PER_ITEM
#endif // EPHOS_LINE_PROCESSING

// TODO: evaluate a way to compare bit count
#if EPHOS_DISTANCES_PER_PACKET == 1
typedef char DistancePacket;
#elif EPHOS_DISTANCES_PER_PACKET == 8
typedef char DistancePacket;
#elif EPHOS_DISTANCES_PER_LOCATION == 16
typedef short DistancePacket;
#elif EPHOS_DISTANCES_PER_LOCATION == 32
typedef int DistancePacket;
#else
#error "Invalid distance packet size"
#endif

// atomic access only availble on int types
// use this if required
#if defined(EPHOS_ATOMICS) && !defined(EPHOS_LINE_PROCESSING)
typedef int Processed;
#else
typedef char Processed;
#endif

typedef struct {
	double radius;
	int cloudSize;
	int lineLength;
} RadiusSearchInfo;

typedef struct  {
    float x,y,z;
} Point;


/**
 * Computes the pairwise squared distances. Results are stored in a matrix.
 * An entry of that matrix indicates whether the distance of the two described cloud
 * is less or equal to the reference distance.
 * cloud: cloud for which we need pairwise distances with size N
 * sqr_distances: resulting distance matrix of size N*N
 * cloudSize: distances calculated per work item
 * radius: reference distance
 */
__kernel void distanceMatrix(
	__global const Point* restrict cloud,
	__global bool*        restrict distances,
	int cloudSize,
	double radius) {
#ifdef EPHOS_LINE_PROCESSING
	int n = cloudSize;
	int j = get_global_id(0);

	if (j < cloudSize) {
		for (int i = 0; i < cloudSize; i++)
		{
			float dx = cloud[i].x - cloud[j].x;
			float dy = cloud[i].y - cloud[j].y;
			float dz = cloud[i].z - cloud[j].z;
			int array_index = i*cloudSize + j;
			distances[array_index] = ((dx*dx + dy*dy + dz*dz) <= radius);
		}
	}
#else // !EPHOS_LINE_PROCESSING

#endif // !EPHOS_LINE_PROCESSING
}

