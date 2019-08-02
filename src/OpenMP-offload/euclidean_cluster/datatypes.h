/**
 * Author:  Florian Stock, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2018
 * License: Apache 2.0 (see attachached File)
 */
#ifndef DATATYPES_H
#define DATATYPES_H

#include <vector>

typedef struct  {
    float x,y,z;
} Point;

typedef struct  {
    double x,y,z;
} PointDouble;

typedef struct {
    float x,y;
} Point2D;


typedef struct {
    double x,y,z,w;
} Orientation;

typedef struct {
    float x, y, z;
    uint8_t r,g,b;
} PointRGB;


typedef struct {
    PointDouble position;
    Orientation orientation;
    PointDouble dimensions;
} Boundingbox;

typedef struct{
    Point2D center;
    Point2D size;
    float angle;
} RotatedRect;

typedef std::vector<Point> PointCloud;
typedef std::vector<PointRGB> PointCloudRGB;

typedef struct {
    std::vector<PointDouble> points;
} Centroid;


typedef struct {
   std::vector<Boundingbox> boxes;
} BoundingboxArray;


typedef struct PointIndices {
    std::vector<int> indices;
} PointIndices;

#define PI 3.1415926535897932384626433832795

#endif


