/**
 * Author:  Florian Stock, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2018
 * Author:  Thilo Gabel, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2020
 * License: Apache 2.0 (see attached files)
 */
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>

#include "points2image.h"

// image storage
// expected image size 800x600 pixels with 4 components per pixel
// __device__ __managed__ float result_buffer[800*600*5];

points2image::points2image() : points2image_base() {}
points2image::~points2image() {}


/**
 * Improvised atomic min/max functions for floats in Cuda
 * Does only work for positive values.
 */
__device__ __forceinline__ float atomicFloatMin(float* addr, float value) {
	return __int_as_float(atomicMin((int*)addr, __float_as_int(value)));
}
__device__ __forceinline__ float atomicFloatMax(float* addr, float value) {
	return __int_as_float(atomicMax((int*)addr, __float_as_int(value)));
}

/**
 * Transforms each point in the point cloud according
 * to the given transformation informaton and
 * writes the results to pixels for postprocessing
 * on the host device.
 * pointcloud: points to transform
 * pixels: transformation results
 * pixelNo: result counter
 * info: transformation and point cloud information
 */
__global__ void computeCloudTransformation(
	const float* __restrict__ pointcloud,
	TransformPixel* __restrict__ pixels,
	int* pixelNo,
	const TransformationInfo info) {

	int iPoint = blockIdx.x*blockDim.x + threadIdx.x;
	TransformPixel pixel = { NULL };
	bool isInside = (iPoint < info.pointNo);
	if (isInside) {
		float* fp = (float *)((uintptr_t)pointcloud + iPoint*info.pointStep);
		Mat13 point2 = {
			fp[0], fp[1], fp[2]
		};
		// apply rotation and translation
		Mat13 point;
		for (int row = 0; row < 3; row++) {
			point.data[row] = info.invTranslation.data[row];
			for (int col = 0; col < 3; col++)
			point.data[row] += point2.data[col]*info.invRotation.data[row][col];
		}
		isInside = (point.data[2] > 2.5);
		if (isInside) {
			// second transformation step
			double pointX = point.data[0]/point.data[2];
			double pointY = point.data[1]/point.data[2];
			double sqrRadius = pointX*pointX + pointY*pointY;
			double distortion = 1.0 + info.distCoeff.data[0]*sqrRadius
				+ info.distCoeff.data[1]*sqrRadius*sqrRadius
				+ info.distCoeff.data[4]*sqrRadius*sqrRadius*sqrRadius;

			double imgX = pointX*distortion
				+ 2.0*info.distCoeff.data[2]*pointX*pointY
				+ info.distCoeff.data[3]*(sqrRadius + 2.0*pointX*pointX);
			double imgY = pointY*distortion
				+ info.distCoeff.data[2]*(sqrRadius + 2.0*pointY*pointY)
				+ 2.0*info.distCoeff.data[3]*pointX*pointY;
			// apply camera intrinsics to yield a point in the image
			pixel.posX = int(info.camScale.data[0]*imgX + info.camOffset.data[0]);
			pixel.posY = int(info.camScale.data[1]*imgY + info.camOffset.data[1]);

			isInside = (0 <= pixel.posX && pixel.posX < info.imageSize.width &&
						0 <= pixel.posY && pixel.posY < info.imageSize.height);
			// safe point characteristics in the image
			if (isInside) {
				pixel.depth = float(point.data[2]*100.0);
				pixel.intensity = fp[4];
			}
		}
	}
	__shared__ int b_pixelSort;
	__shared__ int b_pixelStart;
	// count the number of pixels to allocate in the result buffer
	// and distribute the space to the threads in a block
	int pixNo = __syncthreads_count(isInside);
	if (pixNo > 0) {
		if (threadIdx.x == 0) {
			b_pixelSort = 0;
			b_pixelStart = atomicAdd(pixelNo, pixNo);
		}
		__syncthreads();
		if (isInside) {
			int iPixel = b_pixelStart + atomicAdd(&b_pixelSort, 1);
			pixels[iPixel] = pixel;
		}
	}
}



/**
 * This code is extracted from Autoware, file:
 * ~/Autoware/ros/src/sensing/fusion/packages/points2image/lib/points_image/points_image.cpp
 * It uses the test data that has been read before and applies the algorithm.
 * pointcloud: cloud of points to transform
 * cameraExtrinsicMat: camera matrix used for transformation
 * cameraMat: camera matrix used for transformation
 * distCoeff: distance coefficients for cloud transformation
 * imageSize: the size of the resulting image
 * returns: the two dimensional image of transformed points
 */
PointsImage points2image::cloud2Image(
	PointCloud& pointcloud,
	Mat44& cameraExtrinsicMat,
	Mat33& cameraMat,
	Vec5& distCoeff,
	ImageSize& imageSize) {

	// prepare output structure
	int imagePixelNo = imageSize.width*imageSize.height;
	int pointNo = pointcloud.width*pointcloud.height;
	PointsImage msg;
	msg.max_y = -1;
	msg.min_y = imageSize.height;
	msg.image_height = imageSize.height;
	msg.image_width = imageSize.width;
	msg.intensity = new float[imagePixelNo];
	msg.distance = new float[imagePixelNo];
	msg.min_height = new float[imagePixelNo];
	msg.max_height = new float[imagePixelNo];
	std::memset(msg.intensity, 0, sizeof(float)*imagePixelNo);
	std::memset(msg.distance, 0, sizeof(float)*imagePixelNo);
	std::memset(msg.min_height, 0, sizeof(float)*imagePixelNo);
	std::memset(msg.max_height, 0, sizeof(float)*imagePixelNo);

	// preprocess the given matrices
	Mat33 invR;
	Mat13 invT;
	// transposed 3x3 camera extrinsic matrix
	for (int row = 0; row < 3; row++)
		for (int col = 0; col < 3; col++)
			invR.data[row][col] = cameraExtrinsicMat.data[col][row];
	// translation vector: (transposed camera extrinsic matrix)*(fourth column of camera extrinsic matrix)
	for (int row = 0; row < 3; row++) {
		invT.data[row] = 0.0;
		for (int col = 0; col < 3; col++)
			invT.data[row] -= invR.data[row][col] * cameraExtrinsicMat.data[col][3];
	}
	// call the transformation kernel
	TransformationInfo computeInfo = {
		invR,
		distCoeff,
		invT,
		{ cameraMat.data[0][2] + 0.5, cameraMat.data[1][2] + 0.5 },
		{ cameraMat.data[0][0], cameraMat.data[1][1] },
		pointNo,
		pointcloud.point_step,
		imageSize
	};
	dim3 computeThreadDim(EPHOS_KERNEL_BLOCK_SIZE);
	dim3 computeBlockDim((pointNo + EPHOS_KERNEL_BLOCK_SIZE - 1)/EPHOS_KERNEL_BLOCK_SIZE);
	TransformPixel* arrivingPixels;
	int* arrivingPixelCountBuffer;
	int arrivingPixelNo = 0;

	cudaMallocManaged(&arrivingPixels, sizeof(TransformPixel)*pointNo);
	cudaMalloc(&arrivingPixelCountBuffer, sizeof(int));
	cudaMemcpy(arrivingPixelCountBuffer, &arrivingPixelNo, sizeof(int), cudaMemcpyHostToDevice);
	computeCloudTransformation<<<computeBlockDim, computeThreadDim>>>(
		pointcloud.data, arrivingPixels, arrivingPixelCountBuffer, computeInfo);
	cudaMemcpy(&arrivingPixelNo, arrivingPixelCountBuffer, sizeof(int), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	// postprocess transformation results
	// write pixels to image result
	for (int i = 0; i < arrivingPixelNo; i++) {
		int iPixel = arrivingPixels[i].posY*imageSize.width + arrivingPixels[i].posX;
		if (msg.distance[iPixel] == 0.0 ||
			msg.distance[iPixel] >= arrivingPixels[i].depth) {

			// make the result always deterministic and independent from the point order
			// in case two points get the same distance, take the one with higher intensity
			if ((msg.distance[iPixel] == arrivingPixels[i].depth && msg.intensity[iPixel] < arrivingPixels[i].intensity) ||
				(msg.distance[iPixel] > arrivingPixels[i].depth) ||
				(msg.distance[iPixel] == 0)) {

				msg.intensity[iPixel] = arrivingPixels[i].intensity;
			}
			msg.distance[iPixel] = arrivingPixels[i].depth;
			msg.min_height[iPixel] = -1.25f;

			if (arrivingPixels[i].posY > msg.max_y) {
				msg.max_y = arrivingPixels[i].posY;
			}
			if (arrivingPixels[i].posY < msg.min_y) {
				msg.min_y = arrivingPixels[i].posY;
			}
		}
	}
	cudaFree(arrivingPixels);
	cudaFree(arrivingPixelCountBuffer);
	return msg;
}


