/**
 * Author:  Leonardo Solis, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2018
 * Author:  Thilo Gabel, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2019
 * License: Apache 2.0 (see attachached File)
 */
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
	__global PixelData* restrict g_imagePixels,
	__global int* restrict g_imagePointNo,
	__local int* restrict l_imagePointNo,
	__local int* restrict l_imagePointStart)
{
#ifdef EPHOS_KERNEL_ATOMICS
	// initialize local variables
	if (get_local_id(0) == 0) {
		*l_imagePointNo = 0;
		*l_imagePointStart = -1;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
#endif
	// transformation for one point in the cloud
	const int y = 0;
	int iCloud = (int)get_global_id(0);
	PixelData pixel = {
		{ -1, -1 },
		-1.0,
		-1.0
	};
	int startIndex = -1;

	// discard the calculations for excess elements
	if (iCloud < transformInfo.cloudPointNo) {
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
			double imgx = point2[0] * tmpdist
						+ 2 * transformInfo.distCoeff[2] * point2[0] * point2[1]
						+ transformInfo.distCoeff[3] * (r2 + 2 * point2[0] * point2[0]);
			double imgy = point2[1] * tmpdist
						+ transformInfo.distCoeff[2] * (r2 + 2 * point2[1] * point2[1])
						+ 2 * transformInfo.distCoeff[3] * point2[0] * point2[1];
			// apply the third transformation
			pixel.position[0] = (int)(transformInfo.imageScale[0]*imgx + transformInfo.imageOffset[0]);
			pixel.position[1] = (int)(transformInfo.imageScale[1]*imgy + transformInfo.imageOffset[1]);

			// output points inside image bounds
			if((0 <= pixel.position[0]) && (pixel.position[0] < transformInfo.imageSize[0]) && (0 <= pixel.position[1]) && (pixel.position[1] < transformInfo.imageSize[1])) {
#ifdef EPHOS_KERNEL_ATOMICS
				// determine local start index for dense result writing
				startIndex = atomic_inc(l_imagePointNo);
#else // !EPHOS_KERNEL_ATOMICS
				startIndex = 1;
#endif // !EPHOS_KERNEL_ATOMICS
				pixel.depth = point2[2];
				pixel.intensity = g_cloudPoints[offset1 + 4];
			}
		}
#ifndef EPHOS_KERNEL_ATOMICS
		// write result for every point
		int iPixel = iCloud;
		if (startIndex > -1) {
			g_imagePixels[iPixel] = pixel;
		} else {
			// -1 to disable this entry
			g_imagePixels[iPixel].position[0] = -1;
		}
#endif // !EPHOS_KERNEL_ATOMICS
	}
#ifdef EPHOS_KERNEL_ATOMICS
	// write result densely
	barrier(CLK_LOCAL_MEM_FENCE);
	if (get_local_id(0) == 0) {
		if (*l_imagePointNo > 0) {
			*l_imagePointStart = atomic_add(g_imagePointNo, *l_imagePointNo);
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	if (startIndex > -1) {
		int iPixel = *l_imagePointStart + startIndex;
		g_imagePixels[iPixel] = pixel;
	}
#endif // EPHOS_KERNEL_ATOMICS
}
