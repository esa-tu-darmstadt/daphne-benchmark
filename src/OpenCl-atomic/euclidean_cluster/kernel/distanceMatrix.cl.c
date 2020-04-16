/**
 * Author:  Leonardo Solis, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2018
 * Author:  Thilo Gabel, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2020
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
#if EPHOS_DISTANCES_PER_PACKET == 1 || EPHOS_DISTANCES_PER_PACKET == 8
typedef uchar DistancePacket;
#elif EPHOS_DISTANCES_PER_PACKET == 16
typedef ushort DistancePacket;
#elif EPHOS_DISTANCES_PER_PACKET == 32
typedef uint DistancePacket;
#elif EPHOS_DISTANCES_PER_PACKET == 64
typedef ulong DistancePacket;
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
	int sourceCloudSize;
	int alignedCloudSize;
	int distanceLineLength;
	int queueStartIndex;
	int staticQueueSize;
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
	__global Processed* restrict processed,
	RadiusSearchInfo searchInfo) {
//	int cloudSize,
//	double radius) {
#ifdef EPHOS_LINE_PROCESSING
	int j = get_global_id(0);

	// pivot element with index j
	if (j < searchInfo.sourceCloudSize) {
		// go through one line
		// line processing ignores packets per work item
		// because it always generates a whole line of distance packets
		// step over previously processed cloud elements
		for (int i = 0; i*EPHOS_DISTANCES_PER_PACKET < searchInfo.alignedCloudSize; i++)
		{
			//DistancePacket dist = ~0x0;
			DistancePacket dist = 0x0;
			// build distance packet
			// one packet consists of the nearness indicators for one or more pairwise distances
			for (int k = 0; k < EPHOS_DISTANCES_PER_PACKET; k++) {
				// distabled because of misaligned address error
				//__global const float3* pCloud1 = (__global const float3*)&cloud[i*EPHOS_DISTANCES_PER_PACKET + k];
				//__global const float3* pCloud2 = (__global const float3*)&cloud[j];
				//float3 d = (*pCloud1) - (*pCloud2); //cloud[i*EPHOS_DISTANCES_PER_PACKET + k] - cloud[j];
				float dx = cloud[i*EPHOS_DISTANCES_PER_PACKET + k].x - cloud[j].x;
				float dy = cloud[i*EPHOS_DISTANCES_PER_PACKET + k].y - cloud[j].y;
				float dz = cloud[i*EPHOS_DISTANCES_PER_PACKET + k].z - cloud[j].z;
				//if (dx*dx + dy*dy + dz*dz > searchInfo.radius) {
				//	dist ^= (0x1<<k);
				//}
				if (dx*dx + dy*dy + dz*dz <= searchInfo.radius) {
				//if (d.x*d.x + d.y*d.y + d.z*d.z <=radius) {
					dist |= (0x1<<k);
				}
			}
			//int iDist = i*searchInfo.distanceLineLength + j; // spaced out work group write
			int iDist = j*searchInfo.distanceLineLength + i;
			distances[iDist] = dist;
			//float dx = cloud[i].x - cloud[j].x;
			//float dy = cloud[i].y - cloud[j].y;
			//float dz = cloud[i].z - cloud[j].z;
			//int array_index = i*cloudSize + j;
			//distances[array_index] = ((dx*dx + dy*dy + dz*dz) <= radius);
		}
		processed[j] = 0x0;
	} else if (j < searchInfo.alignedCloudSize) {
		processed[j] = 0x1;
	}
#endif // EPHOS_LINE_PROCESSING
}

