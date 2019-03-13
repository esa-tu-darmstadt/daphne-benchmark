#ifndef datatypes_h
#define datatypes_h

#include <vector>

// 3d euclidean point with intensity
typedef struct PointXYZI {
    float data[4];
} PointXYZI;

typedef struct Matrix4f {
  float data[4][4];
} Matrix4f;

typedef struct Mat33 {
  double data[3][3];
} Mat33;

typedef struct Mat66 {
  double data[6][6];
} Mat66;

typedef struct Mat36 {
  double data[3][6];
} Mat36;

typedef struct Mat186 {
  double data[18][6];
} Mat186;


typedef struct Vec5 {
  double data[5];
} Vec5;

typedef double Vec3[3];

typedef double Vec6[6];

typedef struct Point4d {
    float x,y,z,i;
} Point4d;

typedef std::vector<PointXYZI> PointCloudSource;
typedef PointCloudSource PointCloud;

typedef struct CallbackResult {
    bool converged;
    Matrix4f final_transformation;
    double fitness_score;
} CallbackResult;

typedef struct TargetGridLeafConstPtr {
    Mat33 invCovariance;
    Vec3 mean;
    int numberPoints;
    int voxel_x;
    int voxel_y;
    int voxel_z;
} TargetGridLeadConstPtr;

typedef std::vector<TargetGridLeafConstPtr> VoxelGrid;

Matrix4f Matrix4f_Identity = { .data = {{1.0, 0.0, 0.0, 0.0}, {0.0, 1.0, 0.0, 0.0}, {0.0, 0.0, 1.0, 0.0}, {0.0, 0.0, 0.0, 1.0}} };

#define PI 3.1415926535897932384626433832795

#endif

