/**
 * Radius search base on a precomputed distance matrix.
 * Near points are marked through the indices array.
 * The search can be executed for multiple reference points.
 * 
 * seedQueue: indices of reference points
 * distances: precomputed distance matrix
 * iQueueStart: point index to start at
 * staticQueueSize: point index to end at
 * cloudSize: number of elements in the point cloud
 */
__kernel
void __attribute__ ((reqd_work_group_size(NUMWORKITEMS_PER_WORKGROUP,1,1)))
parallelRadiusSearch(
	__global int*  restrict seedQueue,
	__global const bool* restrict distances,
	__global int* processed,
	__global int* nextQueueSize,
	int iQueueStart,
	int staticQueueSize,
	int cloudSize)
{
	int id = get_global_id(0);
	if (id < cloudSize && !processed[id]) {
		bool found = false;
		bool is_skipped = false;
		for (int iQueue = iQueueStart; iQueue < staticQueueSize; iQueue++)
		{
			if (id == seedQueue[iQueue])
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
				int array_index = seedQueue[iQueue] * cloudSize+id;;
				if (distances[array_index])
					found = true;
			}
		}
		if (found) {
			int iTarget = atomic_inc(nextQueueSize);
			seedQueue[iTarget] = id;
			processed[id] = 1;
		}
	}
}


