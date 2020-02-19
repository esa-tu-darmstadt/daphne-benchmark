/**
 * Author:  Leonardo Solis, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2018
 * Author:  Thilo Gabel, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2019
 * License: Apache 2.0 (see attachached File)
 */
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
__kernel void radiusSearch(
	__global int*  restrict seedQueue,
	__global const bool* restrict distances,
	__global bool* processed,
	__global int* nextQueueSize,
	int iQueueStart,
	int staticQueueSize,
	int cloudSize)
{
#ifdef EPHOS_LINE_PROCESSING
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
			processed[id] = true;
		}
	}
#else

#endif
}


