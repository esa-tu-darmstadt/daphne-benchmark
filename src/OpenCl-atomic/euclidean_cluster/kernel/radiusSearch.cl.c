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
	__global const DistancePacket* restrict distances,
	__global bool* processed,
	__global int* nextQueueSize,
	int iQueueStart,
	int staticQueueSize,
	int cloudSize)
{
#ifdef EPHOS_LINE_PROCESSING
	int iPivotPacket = get_global_id(0);
	// the number of cloud elements to compare is determined by the number of distances per distance packet
	int iPivotStart = iPivotPacket*EPHOS_DISTANCES_PER_PACKET;
	if (iPivotStart < cloudSize && !processed[iPivotStart]) {
		DistancePacket skipped = 0;
		int findNo = EPHOS_DISTANCES_PER_PACKET;
		for (int i = 0; i < EPHOS_DISTANCES_PER_PACKET; i++) {
			// mark already processed elements as skipped
			// skipped elements are removed from find
			if (processed[iPivotStart + i]) {
				skipped |= 0x1<<i;
				findNo -= 1;
			}
		}
		// only process further if there are non processed elements
		if (findNo > 0) {
			DistancePacket find = 0;
			findNo = 0;
			// line processing ignores packets per work item
			// as it compares a constant number of cloud elements with all new cluster elements
			for (int iQueue = iQueueStart; iQueue < staticQueueSize; iQueue++)
			{
				int iDist = seedQueue[iQueue]*cloudSize + iPivotPacket;
				DistancePacket dist = distances[iDist];
				for (int i = 0; i < EPHOS_DISTANCES_PER_PACKET; i++) {
					if ((skipped>>i & 0x1) == 0) {
						// skip elements that are already in queue
						if (iPivotPacket + i == seedQueue[iQueue]) {
							// mark as skipped and remove from find
							skipped |= 0x1<<i;
							// disable element find status
							find &= 0x1<<i ^ ~0x0;
							// subtract previously added occurrence
							// this subtracts 1 if find status is set
							findNo -= find>>i & 0x1;
						} else {
							// add elements that are near the current queue element
							//int iDist = seedQueue[iQueue]*cloudSize + iPivotPacket + i;
							if ((dist>>i & 0x1) == 1) {
								// add a new occurence
								// this adds 0 if find state is already set
								findNo += find>>i ^ 0x1;
								// mark the element as found
								find |= 0x1<<i;
							}
						}
					}
				}
			}
			if (findNo > 0) {
				//findNo = 1;
				int iTarget = atomic_add(nextQueueSize, findNo);
				for (int i = 0; i < EPHOS_DISTANCES_PER_PACKET; i++) {
					// add elements that have find status set and skipped status unset
					if ((find>>i & 0x1) != 0 && (skipped>>i & 0x1) == 0) {
						seedQueue[iTarget] = iPivotStart + i;
						processed[iPivotStart + i] = true;
					}
				}
			}
		}
	}






/*

			int seedElement = seedQueue[iQueue];
			int iPacket = seedElement*cloudSize + iPivotPacket;
			DistancePacket dist = distances[iPacket];
			// go through one packet
			for (int i = 0; i < EPHOS_DISTANCES_PER_PACKET; i++) {
				// skip elements already in cluster
				if (iPivotStart + i == seedElement) {
					skipped |= 0x1<<i;
				}
				// near elements to queue
				if ((dist & (0x1<<i)) != 0 && (skipped & (0x1<<i)) != 0) {
					found |= 0x1<<i;
				}
			}


// 			if (iPivotPacket == seedQueue[iQueue])
// 			{
// 				found = true;
// 				is_skipped = true;
// 			}
// 			else
// 			{
// 				is_skipped = false;
// 			}
// 			if (!is_skipped)
// 			{
// 				int array_index = seedQueue[iQueue] * cloudSize+iPivotPacket;
// 				if (distances[array_index] != 0)
// 					found = true;
// 			}
		}
		if (found != 0) {
			int foundNo = 0;
			for (int i = 0; i < EPHOS_DISTANCES_PER_PACKET; i++) {
				if ((found & (0x1<<i)) != 0 && (skipped & (0x1<<i)) == 0) {
					foundNo += 1;
				}
			}
			int iTarget = atomic_add(nextQueueSize, foundNo);
			for (int i = 0; i < EPHOS_DISTANCES_PER_PACKET; i++) {
				if ((found & (0x1<<i)) != 0 && (skipped & (0x1<<i)) == 0) {
					seedQueue[iTarget++] = iPivotStart + i;
					processed[iPivotStart + i] = true;

				}
			}
// 			seedQueue[iTarget] = iPivotPacket;
// 			processed[iPivotPacket] = true;
		}
	}*/
#else

#endif
}


