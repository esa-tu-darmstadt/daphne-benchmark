/**
 * Author:  Leonardo Solis, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2018
 * Author:  Thilo Gabel, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2019
 * License: Apache 2.0 (see attachached File)
 */
#if defined(EPHOS_KERNEL_TRANSFORMS_PER_ITEM) && (EPHOS_KERNEL_TRANSFORMS_PER_ITEM < 2)
#undef EPHOS_KERNEL_TRANSFORMS_PER_ITEM
#endif

#if !defined(EPHOS_KERNEL_ATOMICS) && defined(EPHOS_KERNEL_LOCAL_ATOMICS)
#undef EPHOS_KERNEL_LOCAL_ATOMICS
#endif

typedef struct TransformInfo {
	double initRotation[3][3];
	double initTranslation[3];
	double imageScale[2];
	double imageOffset[2];
	double distCoeff[5];
	int imageSize[2];
	int cloudPointNo;
	int cloudPointStep;
} TransformInfo;

typedef struct PixelData {
	int position[2];
	float depth;
	float intensity;
} PixelData;

typedef float PointData;

__kernel void pointcloud2_to_image(
	const TransformInfo transformInfo,
	__global const PointData* restrict g_cloudPoints,
	__global PixelData* restrict g_imagePixels
#ifdef EPHOS_KERNEL_ATOMICS
	,__global int* restrict g_imagePointNo
#ifdef EPHOS_KERNEL_LOCAL_ATOMICS
	,__local int* restrict l_imagePointNo,
	__local int* restrict l_imagePointStart
#endif // EPHOS_KERNEL_LOCAL_ATOMICS
#endif // EPHOS_KERNEL_ATOMICS
) {
#if defined(EPHOS_KERNEL_ATOMICS) && defined(EPHOS_KERNEL_LOCAL_ATOMICS)
	// initialize local variables
	if (get_local_id(0) == 0) {
		*l_imagePointNo = 0;
		*l_imagePointStart = -1;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
#endif // EPHOS_KERNEL_ATOMICS && EPHOS_KERNEL_LOCAL_ATOMICS

	// transformation for one point in the cloud
	int iCloudStart = (int)get_global_id(0);

	// temporary pixel data
#ifdef EPHOS_KERNEL_LOCAL_ATOMICS
#ifdef EPHOS_KERNEL_TRANSFORMS_PER_ITEM
	PixelData pixel[EPHOS_KERNEL_TRANSFORMS_PER_ITEM];
	int interPixelNo = 0;
#else // !EPHOS_KERNEL_TRANSFORMS_PER_ITEM
	PixelData pixel[1];
#endif // !EPHOS_KERNEL_TRANSFORMS_PER_ITEM
	int iStart = -1;
#endif // EPHOS_KERNEL_LOCAL_ATOMICS

#ifndef EPHOS_KERNEL_ATOMICS
	bool arriving = false;
#endif // EPHOS_KERNEL_ATOMICS

#ifdef EPHOS_KERNEL_TRANSFORMS_PER_ITEM
	// iterate over assigned range
	int iCloudOffset = 0;
	for (int iCloud = iCloudStart*EPHOS_KERNEL_TRANSFORMS_PER_ITEM;
		 iCloud < transformInfo.cloudPointNo && iCloudOffset < EPHOS_KERNEL_TRANSFORMS_PER_ITEM;
		 iCloud++, iCloudOffset++) {
		{
#else // !EPHOS_KERNEL_TRANSFORMS_PER_ITEM
	// discard the calculations for excess elements
	if (iCloudStart < transformInfo.cloudPointNo) {
		int iCloud = iCloudStart;
		{
# endif
			int offset1 = iCloud*transformInfo.cloudPointStep;
			double point1[3];
			point1[0] = g_cloudPoints[offset1 + 0];
			point1[1] = g_cloudPoints[offset1 + 1];
			point1[2] = g_cloudPoints[offset1 + 2];
			// apply the initial transformation
			// for (int row = 0; row < 3; row++) {
				// point2.data[row] = transformInfo.initTranslation[row];
				// for (int col = 0; col < 3; col++)
					// point2.data[row] += point1[col] * transformInfo.initRotation[row][col];
			// }
			double point2[3];
			point2[0] = transformInfo.initTranslation[0] +
				point1[0]*transformInfo.initRotation[0][0] +
				point1[1]*transformInfo.initRotation[0][1] +
				point1[2]*transformInfo.initRotation[0][2];
			point2[1] = transformInfo.initTranslation[1] +
				point1[0]*transformInfo.initRotation[1][0] +
				point1[1]*transformInfo.initRotation[1][1] +
				point1[2]*transformInfo.initRotation[1][2];
			point2[2] = transformInfo.initTranslation[2] +
				point1[0]*transformInfo.initRotation[2][0] +
				point1[1]*transformInfo.initRotation[2][1] +
				point1[2]*transformInfo.initRotation[2][2];

			// discard elements with low depth after transformation
			if (point2[2] > 2.5) {
				// perspective division
				point2[0] = point2[0]/point2[2];
				point2[1] = point2[1]/point2[2];
				// apply the distortion coefficients
				double r2 = point2[0] * point2[0] + point2[1] * point2[1];
				double tmpdist = 1 + transformInfo.distCoeff[0] * r2
						+ transformInfo.distCoeff[1] * r2 * r2
						+ transformInfo.distCoeff[4] * r2 * r2 * r2;
				double lensX = point2[0] * tmpdist
							+ 2 * transformInfo.distCoeff[2] * point2[0] * point2[1]
							+ transformInfo.distCoeff[3] * (r2 + 2 * point2[0] * point2[0]);
				double lensY = point2[1] * tmpdist
							+ transformInfo.distCoeff[2] * (r2 + 2 * point2[1] * point2[1])
							+ 2 * transformInfo.distCoeff[3] * point2[0] * point2[1];
				// apply the third transformation
				int imgX = (int)(transformInfo.imageScale[0]*lensX + transformInfo.imageOffset[0]);
				int imgY = (int)(transformInfo.imageScale[1]*lensY + transformInfo.imageOffset[1]);

				// output points inside image bounds
				if ((0 <= imgX) && (imgX < transformInfo.imageSize[0]) &&
					(0 <= imgY) && (imgY < transformInfo.imageSize[1])) {
#ifdef EPHOS_KERNEL_ATOMICS
#ifdef EPHOS_KERNEL_LOCAL_ATOMICS
#ifdef EPHOS_KERNEL_TRANSFORMS_PER_ITEM
					pixel[interPixelNo] = (PixelData){
						{ imgX, imgY },
						point2[2],
						g_cloudPoints[offset1 + 4]
					};
					interPixelNo += 1;
#else // !EPHOS_KERNEL_TRANSFORMS_PER_ITEM
					// local start determination for one points in one work item
					iStart = atomic_inc(l_imagePointNo);
					pixel[0] = (PixelData){
						{ imgX, imgY },
						point2[2],
						g_cloudPoints[offset1 + 4]
					};
#endif // !EPHOS_KERNEL_TRANSFORMS_PER_ITEM

#else // !EPHOS_KERNEL_LOCAL_ATOMICS
					int iStart = atomic_inc(g_imagePointNo);
					g_imagePixels[iStart] = (PixelData){
						{ imgX, imgY },
						point2[2],
						g_cloudPoints[offset1 + 4]
					};
#endif // !EPHOS_KERNEL_LOCAL_ATOMICS
#else // !EPHOS_KERNEL_ATOMICS
					arriving = true;
					g_imagePixels[iCloud] = (PixelData){
						{ imgX, imgY },
						point2[2],
						g_cloudPoints[offset1 + 4]
					};
#endif // !EPHOS_KERNEL_ATOMICS
				}
			}
#ifndef EPHOS_KERNEL_ATOMICS
			// mark with -1 for every point outside image constraints
			if (!arriving) {
				g_imagePixels[iCloud].position[0] = -1;
			}
#endif // !EPHOS_KERNEL_ATOMICS
		}
	}
#if defined(EPHOS_KERNEL_TRANSFORMS_PER_ITEM) && defined(EPHOS_KERNEL_LOCAL_ATOMICS)
	// deferred local start determination for multiple points in one work item
	if (interPixelNo > 0) {
		iStart = atomic_add(l_imagePointNo, interPixelNo);
	}
#endif
#if defined(EPHOS_KERNEL_ATOMICS) && defined(EPHOS_KERNEL_LOCAL_ATOMICS)
	// write result densely
	barrier(CLK_LOCAL_MEM_FENCE);
	if (get_local_id(0) == 0) {
		if (*l_imagePointNo > 0) {
			*l_imagePointStart = atomic_add(g_imagePointNo, *l_imagePointNo);
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
#ifdef EPHOS_KERNEL_TRANSFORMS_PER_ITEM
	for (int i = 0; i < interPixelNo; i++) {
		int iPixel = *l_imagePointStart + iStart + i;
		g_imagePixels[iPixel] = pixel[i];
	}
#else // !EPHOS_KERNEL_TRANSFORMS_PER_ITEM
	if (iStart > -1) {
		int iPixel = *l_imagePointStart + iStart;
		g_imagePixels[iPixel] = pixel[0];
	}
#endif // !EPHOS_KERNEL_TRANSFORMS_PER_ITEM
#endif // EPHOS_KERNEL_ATOMICS && EPHOS_KERNEL_LOCAL_ATOMICS
}
