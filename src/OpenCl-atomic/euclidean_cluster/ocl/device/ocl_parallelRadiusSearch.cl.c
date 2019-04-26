/**
 * Radius search base on a precomputed distance matrix.
 * Near points are marked through the indices array.
 * The search can be executed for multiple reference points.
 * 
 * point_index: indices of reference points
 * indices: near point marks
 * sqr_distances: precomputed distance matrix
 * start_index: point index to start at
 * search_points: point index to end at
 * cloud_size: number of elements in the point cloud
 */
__kernel
void __attribute__ ((reqd_work_group_size(NUMWORKITEMS_PER_WORKGROUP,1,1)))
parallelRadiusSearch(
	__global const int*  restrict point_index,
	__global       bool* restrict indices, 
	__global const bool* restrict sqr_distances, 
	int start_index, 
	int search_points, 
	int cloud_size)
{
	int id = get_global_id(0);
	if (id < cloud_size) {
		bool found = false;
		bool is_skipped = false;
		for (int search_point_index = start_index; search_point_index < search_points; search_point_index++)
		{
			if (id == point_index[search_point_index])
			{
				found = true;
				is_skipped = true;
			} 
			else
			{
				is_skipped = false;
			}
			if (!is_skipped) 
			{
				int array_index = point_index[search_point_index] * cloud_size+id;;
				if ( sqr_distances[array_index])
					found = true;
			}
		}
		indices[id] = found;
	}
}


