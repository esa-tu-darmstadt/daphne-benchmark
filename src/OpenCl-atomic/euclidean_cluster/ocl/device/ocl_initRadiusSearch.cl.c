
typedef struct  {
    float x,y,z;
} Point;


/**
 * Computes the pairwise squared distances. Results are stored in a matrix.
 * An entry of that matrix indicates whether the distance of the two described points 
 * is less or equal to the reference distance.
 * points: points for which we need pairwise distances with size N
 * sqr_distances: resulting distance matrix of size N*N
 * number_points: distances calculated per work item
 * radius_sqr: reference distance
 */
__kernel void 
__attribute__ ((reqd_work_group_size(NUMWORKITEMS_PER_WORKGROUP,1,1)))
initRadiusSearch(
	__global const Point* restrict points,
	__global bool*        restrict sqr_distances,
	int    number_points, 
	#if defined (DOUBLE_FP)
	double radius_sqr
	#else
	float radius_sqr
	#endif
) {
	int n = number_points;
	int j = get_global_id(0);

	if (j < number_points) {
		for (int i = 0; i < number_points; i++)
		{
			float dx = points[i].x - points[j].x;
			float dy = points[i].y - points[j].y;
			float dz = points[i].z - points[j].z;
			int array_index = i*number_points + j;
			sqr_distances[array_index] = ((dx*dx + dy*dy + dz*dz) <= radius_sqr );
		}
		sqr_distances[j*number_points + j] = false; // processed indicator
	}

}

