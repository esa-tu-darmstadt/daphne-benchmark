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
#ifdef EPHOS_LINE_PROCESSING
	// every work item processes its own packet with EPHOS_DISTANCES_PER_PACKET points
	int iPacket = get_global_id(0);
	// the corresponding cloud elements are identified by iCloud +1/+2/+3/...
	int iCloud = get_global_id(0)*EPHOS_DISTANCES_PER_PACKET;

	//if (iPivotStart < searchInfo.sourceCloudSize) {
	if (iCloud < searchInfo.sourceCloudSize) {
		DistancePacket skip = 0x0;
		int findNo = EPHOS_DISTANCES_PER_PACKET;
		for (int p = 0; p < EPHOS_DISTANCES_PER_PACKET; p++) {
			if (processed[iCloud + p]) {
				skip |= 0x1<<p;
				findNo -= 1;
			}
		}
		// only process further if there are non processed elements
		if (findNo > 0) {
		//if (skip != (DistancePacket)~0x0) {
			DistancePacket find = 0x0;
			findNo = 0;
			// line processing ignores packets per work item
			// as it compares a constant number of cloud elements with all new cluster elements
			for (int iQueue = searchInfo.queueStartIndex; iQueue < searchInfo.staticQueueSize; iQueue++)
			{
				int seedElement = seedQueue[iQueue];
				int iDist = seedElement*searchInfo.distanceLineLength + iPacket;
				DistancePacket dist = distances[iDist];
// 				for (int p = 0; p < EPHOS_DISTANCES_PER_PACKET; p++) {
// 					// add near elements
// 					// elements in queue have already been marked as skip above
// 					if (!(skip>>p & 0x1) && (dist>>p & 0x1)) {
// 						//findNo += ((find>>p & 0x1) ^ 0x1);
// 						find |= 0x1<<p;
// 					}
// 				}
				// add near elements
				// but exclude the ones that are already in queue (marked as skip)
				find |= (~skip & dist);
			}
			// first count the number of new points to add
			// note: faster do the simpler operations before and count in the end
			for (int p = 0; p < EPHOS_DISTANCES_PER_PACKET; p++) {
				if (find>>p & 0x1) {
					findNo += 1;
				}
			}
			// then append them to the seed queue
			int iTarget = atomic_add(nextQueueSize, findNo);
			for (int p = 0; p < EPHOS_DISTANCES_PER_PACKET; p++) {
				if ((find>>p & 0x1)) {
					processed[iCloud + p] = true;
					seedQueue[iTarget] = iCloud + p;
					iTarget += 1;
				}
			}
		}
	}
#endif // EPHOS_LINE_PROCESSING
}


