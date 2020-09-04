/**
 * Author:  Florian Stock, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2018
 * License: Apache 2.0 (see attachached File)
 */
#ifndef EPHOS_DATATYPES_H
#define EPHOS_DATATYPES_H

#include <vector>

// typedef struct  {
//     float x,y,z;
// } Point;
//
// typedef struct  {
//     double x,y,z;
// } PointDouble;
//
// typedef struct {
//     float x,y;
// } Point2D;
//
//
// typedef struct {
//     double x,y,z,w;
// } Orientation;
//
// typedef struct {
//     float x, y, z;
//     uint8_t r,g,b;
// } PointRGB;
//
// typedef struct {
//     PointDouble position;
//     Orientation orientation;
//     PointDouble dimensions;
// } Boundingbox;
//
// typedef struct{
//     Point2D center;
//     Point2D size;
//     float angle;
// } RotatedRect;
//
// typedef Point* PlainPointCloud;
// typedef std::vector<PointRGB> ColorPointCloud;
//
// typedef struct {
//     std::vector<PointDouble> points;
// } Centroid;
//
// typedef struct {
//    std::vector<Boundingbox> boxes;
// } BoundingboxArray;
//
// typedef struct PointIndices {
//     std::vector<int> indices;
// } PointIndices;

typedef struct RadiusSearchInfo {
	double radius;
	int sourceCloudSize;
	int alignedCloudSize;
	int distanceLineLength;
	int queueStartIndex;
	int staticQueueSize;
} RadiusSearchInfo;

// TODO: evaluate
// typedef int8_t Processed;
//
// #ifndef EPHOS_KERNEL_DISTANCE_PACKETS_PER_ITEM
// #define EPHOS_KERNEL_DISTANCE_PACKETS_PER_ITEM 1
// #endif
//
// #ifndef EPHOS_KERNEL_DISTANCES_PER_PACKET
// #define EPHOS_KERNEL_DISTANCES_PER_PACKET 8
// #endif
//
// #if EPHOS_KERNEL_DISTANCES_PER_PACKET == 1 || EPHOS_KERNEL_DISTANCES_PER_PACKET == 8
// typedef uint8_t DistancePacket;
// #elif EPHOS_KERNEL_DISTANCES_PER_PACKET == 16
// typedef uint16_t DistancePacket;
// #elif EPHOS_KERNEL_DISTANCES_PER_PACKET == 32
// typedef uint32_t DistancePacket;
// #elif EPHOS_KERNEL_DISTANCES_PER_PACKET == 64
// typedef uint64_t DistancePacket;
// #else
// #error "Invalid distance packet size"
// #endif
//
// #define PI 3.1415926535897932384626433832795

#endif

