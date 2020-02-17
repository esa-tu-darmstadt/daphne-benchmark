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
    #if defined (DOUBLE_FP)
    double x,y,z;
    #else
    float x,y,z;
    #endif
} PointDouble;

typedef struct {
    float x,y;
} Point2D;


typedef struct {
    #if defined (DOUBLE_FP)
    double x,y,z,w;
    #else
    float x,y,z,w;
    #endif
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

typedef Point* PointCloud;
typedef std::vector<PointRGB> ColorPointCloud;

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

