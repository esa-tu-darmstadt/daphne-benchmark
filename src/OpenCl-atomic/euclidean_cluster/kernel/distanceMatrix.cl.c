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
#elif EPHOS_DISTANCES_PER_PACKET == 16
typedef short DistancePacket;
#elif EPHOS_DISTANCES_PER_PACKET == 32
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

typedef float3 Point3;


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
	__global DistancePacket* restrict distances,
	int cloudSize,
	double radius) {
#ifdef EPHOS_LINE_PROCESSING
	int n = cloudSize;
	int j = get_global_id(0);

	if (j < cloudSize) {
		for (int i = 0; i*EPHOS_DISTANCES_PER_PACKET < cloudSize; i++)
		{
			DistancePacket dist = 0;
			for (int k = 0; k < EPHOS_DISTANCES_PER_PACKET; k++) {
				// distabled because of misaligned address error
				//__global const float3* pCloud1 = (__global const float3*)&cloud[i*EPHOS_DISTANCES_PER_PACKET + k];
				//__global const float3* pCloud2 = (__global const float3*)&cloud[j];

				//float3 d = (*pCloud1) - (*pCloud2); //cloud[i*EPHOS_DISTANCES_PER_PACKET + k] - cloud[j];
				float dx = cloud[i*EPHOS_DISTANCES_PER_PACKET + k].x - cloud[j].x;
				float dy = cloud[i*EPHOS_DISTANCES_PER_PACKET + k].y - cloud[j].y;
				float dz = cloud[i*EPHOS_DISTANCES_PER_PACKET + k].z - cloud[j].z;
				if (dx*dx + dy*dy + dz*dz <= radius) {
				//if (d.x*d.x + d.y*d.y + d.z*d.z <=radius) {
					dist |= 0x1 << k;
				}
			}
			int iDist = i*cloudSize + j; // spaced out work group write
			distances[iDist] = dist;
			//float dx = cloud[i].x - cloud[j].x;
			//float dy = cloud[i].y - cloud[j].y;
			//float dz = cloud[i].z - cloud[j].z;
			//int array_index = i*cloudSize + j;
			//distances[array_index] = ((dx*dx + dy*dy + dz*dz) <= radius);
		}
	}
#else // !EPHOS_LINE_PROCESSING

#endif // !EPHOS_LINE_PROCESSING
}

