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
	RadiusSearchInfo searchInfo)
{
//#ifndef EPHOS_LINE_PROCESSING // disabled for testing
#if 0
	int iPivotPacket = get_global_id(0);
	// the number of cloud elements to compare is determined by the number of distances per distance packet
	int iPivotStart = iPivotPacket*EPHOS_DISTANCES_PER_PACKET;
	//if (iPivotStart < searchInfo.sourceCloudSize) {
	if (iPivotStart < searchInfo.alignedCloudSize) {
		DistancePacket skipped = 0x0;
		int findNo = EPHOS_DISTANCES_PER_PACKET;
// 		for (int i = 0; i < EPHOS_DISTANCES_PER_PACKET; i++) {
// 			// mark already processed elements as skipped
// 			// skipped elements are removed from find
// 			if (processed[iPivotStart + i]) {
// 				skipped |= 0x1<<i;
// 				findNo -= 1;
// 			}
// 		}
		// only process further if there are non processed elements
		if (findNo > 0) {
		//if (skipped != (DistancePacket)~0x0) {
			DistancePacket find = 0x0;
			findNo = 0;
			// line processing ignores packets per work item
			// as it compares a constant number of cloud elements with all new cluster elements
			for (int iQueue = searchInfo.queueStartIndex; iQueue < searchInfo.staticQueueSize; iQueue++)
			{
				int seedElement = seedQueue[iQueue];
				int iDist = seedElement*searchInfo.distanceLineLength + iPivotPacket;
				DistancePacket dist = distances[iDist];
				for (int i = 0; i < EPHOS_DISTANCES_PER_PACKET; i++) {
					// skip elements that are already in queue
					// need to compare with seed
					// because host does not mark first element as processed
					// TODO: necessary for correctness?
					if (seedElement == iPivotStart + i) {
						skipped |= 0x1<<i;
					} else if (processed[iPivotStart + i]) {
						skipped |= 0x1<<i;
					} else if ((skipped>>i & 0x1) == 0x0) {
						// skip elements that are already in queue
						//if (iPivotPacket + i == seedElement) {
						if (false) {
							// mark as skipped and remove from find
							skipped |= 0x1<<i;
							// disable element find status
//							find &= 0x1<<i ^ ~0x0;
							// subtract previously added occurrence
							// this subtracts 1 if find status is set
//							findNo -= find>>i & 0x1;
						} else {
							// add elements that are near the current queue element
							//int iDist = seedQueue[iQueue]*cloudSize + iPivotPacket + i;
							if ((dist>>i & 0x1) == 0x1) {
								// add a new occurence
								// adds 1 if the find state is not already set
//								findNo += find>>i ^ 0x1;
								// mark the element as found
								find |= 0x1<<i;
							}
						}
					}
				}
			}
			//if (findNo > 0) {
			if (true) {
				findNo = 0;
				for (int i = 0; i < EPHOS_DISTANCES_PER_PACKET; i++) {
					if ((find>>i & 0x1) == 0x1 && (skipped>>i & 0x1) == 0x0) {
						findNo += 1;
					}
				}
				//int iTarget = atomic_add(nextQueueSize, findNo);
				//int iTarget = atomic_inc(nextQueueSize);
				int iTarget = atomic_add(nextQueueSize, findNo);

				//atomic_dec(nextQueueSize);
				for (int i = 0; i < EPHOS_DISTANCES_PER_PACKET; i++) {
					// add elements that have find status set and skipped status unset
					if ((find>>i & 0x1) == 0x1 && (skipped>>i & 0x1) == 0x0) {
						//seedQueue[iTarget] = iPivotStart + i;
						seedQueue[iTarget++] = iPivotStart + i;
						processed[iPivotStart + i] = true;
					}
				}
			}
		}
	}
#else
	int iPacket = get_global_id(0);
	int iCloud = get_global_id(0)*EPHOS_DISTANCES_PER_PACKET;

	//if (iPivotStart < searchInfo.sourceCloudSize) {
	if (iCloud < searchInfo.sourceCloudSize) {
		DistancePacket skipped = 0x0;
		int findNo = 0;
		// only process further if there are non processed elements
		if (true) {
		//if (skipped != (DistancePacket)~0x0) {
			DistancePacket find = 0x0;
			findNo = 0;
			// line processing ignores packets per work item
			// as it compares a constant number of cloud elements with all new cluster elements
			for (int iQueue = searchInfo.queueStartIndex; iQueue < searchInfo.staticQueueSize; iQueue++)
			{
				int seedElement = seedQueue[iQueue];
				int iDist = seedElement*searchInfo.distanceLineLength + iPacket;
				DistancePacket dist = distances[iDist];
				for (int p = 0; p < EPHOS_DISTANCES_PER_PACKET; p++) {
					// skip elements that are already in queue
					// need to compare with seed
					// because host does not mark first element as processed
					// TODO: necessary for correctness?
					if (processed[iCloud + p]) {
						skipped |= 0x1<<p;
					} else if (seedElement == iCloud + p) {
						skipped |= 0x1<<p;
					} else if (!(skipped>>p & 0x1) && (dist>>p & 0x1)) {
						find |= 0x1<<p;
					}
				}
			}
			if (true) {
				findNo = 0;
				for (int p = 0; p < EPHOS_DISTANCES_PER_PACKET; p++) {
					if (!(skipped>>p & 0x1) && (find>>p & 0x1)) {
						findNo += 1;
					}
				}
				int iTarget = atomic_add(nextQueueSize, findNo);
				for (int p = 0; p < EPHOS_DISTANCES_PER_PACKET; p++) {
					if (!(skipped>>p & 0x1) && (find>>p & 0x1)) {
						processed[iCloud + p] = true;
						seedQueue[iTarget] = iCloud + p;
						iTarget += 1;
					}
				}
			}
		}
	}
#endif
}


