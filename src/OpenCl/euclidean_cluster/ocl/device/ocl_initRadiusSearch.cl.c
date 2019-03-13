
// Copied from euclidean_cluster/datatypes.h

// just a 3D point (in pcl PointXYZ)
typedef struct  {
    float x,y,z;
} Point;


/**
   Precomputes all distances and stores the results in sqr_distances.
   Due to symmetry of the distance function (a to b as far apart as b to a),
   only one is stored.
   Size of the aray is (N*(N-1))/2.
   Distance of point i to point j (0..N-1) is stored at index:
       i == j?  nothing stored, distance is 0
       i  > j?  nothing stored, distance is equal to j, i
       j <  i?  distance is stored at:  (((i-1) * i)/2) + j
       To save computations, only the squared distance is computed
   (sufficient for comparison, does not require square root)
   and it is only stored wether it is smaller than a given radius or not
*/
__kernel
void __attribute__ ((reqd_work_group_size(NUMWORKITEMS_PER_WORKGROUP,1,1)))
initRadiusSearch(__global const Point* restrict points,
		 __global bool*        restrict sqr_distances,
                 int    number_points, 
                 #if defined (DOUBLE_FP)
                 double radius_sqr
                 #else
                 float radius_sqr
                 #endif
)
{
  int n = number_points;

/*
  #if defined (DOUBLE_FP)
  double radius_sqr = radius * radius;
  #else
  float radius_sqr = radius * radius;
  #endif
*/

  //int j = blockIdx.x * blockDim.x + threadIdx.x;
  int j = get_global_id(0);

  if (j >= number_points) {
    // Not supported in OpenCL
    // return;
  }
  else {
    for (int i = 0; i < n; i++)
        {
	  float dx = points[i].x - points[j].x;
	  float dy = points[i].y - points[j].y;
	  float dz = points[i].z - points[j].z;
	  int array_index = i*n + j; //(((i-1) * i)/2) +j;
	  sqr_distances[array_index] = ((dx*dx + dy*dy + dz*dz) <= radius_sqr );
        }
  }
}

