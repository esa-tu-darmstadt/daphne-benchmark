/**
 * Author:  Leonardo Solis, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2018
 * License: Apache 2.0 (see attachached File)
 */
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
__kernel void 
__attribute__ ((reqd_work_group_size(NUMWORKITEMS_PER_WORKGROUP,1,1)))
initRadiusSearch(
	__global const Point* restrict cloud,
	__global bool*        restrict distances,
	int cloudSize,
	#if defined (DOUBLE_FP)
	double radius
	#else
	float radius
	#endif
) {
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
}

