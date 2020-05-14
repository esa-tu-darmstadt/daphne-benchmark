/**
 * Author:  Florian Stock, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2018
 * License: Apache 2.0 (see attached files)
 */
#ifndef datatypes_h
#define datatypes_h

// from the task often unordered sets would make
// more sense, but for the benchmark, we want output
// to be guaranteed to be always in the same order
// (for comparison of the results)
#include <vector>

// just a 3D point (in pcl PointXYZ)
typedef struct  {
    float x,y,z;
} Point;

typedef struct  {
    double x,y,z;
} PointDouble;


// a 2D Point (in CV a Point2f)
typedef struct {
    float x,y;
} Point2D;


typedef struct {
    double x,y,z,w;
} Orientation;
    
// point and additional RGB  value
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
    Point2D center; //< the rectangle mass center
    Point2D size;    //< width and height of the rectangle
    float angle;    //< the rotation angle. When the angle is 0, 90, 180, 270 etc., the rectangle becomes an up-right rectangle.
} RotatedRect;
    
// PointCloud<type>
// need: iterator, add element, add all elements of other point cloud (use vector?)

typedef Point* PointCloud;
typedef std::vector<PointRGB> PointCloudRGB;
    
//centroid: set of points, push_back
typedef struct {
    std::vector<PointDouble> points;
} Centroid;

// BoundingBoxArray basically an array of bounding boxes
typedef struct {
   std::vector<Boundingbox> boxes;
} BoundingboxArray;

//pointIndices: vector of int
typedef struct PointIndices {
    std::vector<int> indices;
} PointIndices;



// some CV macros
#define CMP(a,b) (((a) > (b)) - ((a) < (b)))
#define SIGN(a) CMP((a),0)
#define PI 3.1415926535897932384626433832795
#define MIN(a,b) ((a>b)? (b) : (a))
#define MAX(a,b) ((a<b)? (b) : (a))

#endif

