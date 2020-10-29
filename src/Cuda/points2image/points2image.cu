/**
 * Author:  Florian Stock, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2018
 * License: Apache 2.0 (see attached files)
 */
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>

#include "points2image.h"

// maximum allowed deviation from the reference results
#define MAX_EPS 0.001
// number of GPU threads
#define THREADS 256

// image storage
// expected image size 800x600 pixels with 4 components per pixel
// TODO reduce to 4 components again
__device__ __managed__ float result_buffer[800*600*5];
//__device__ __managed__ float result_buffer[800*600*4];

points2image::points2image() : points2image_base() {}
points2image::~points2image() {}

/**
 * Reads the next point cloud
 */
void  points2image::parsePointCloud(std::ifstream& input_file, PointCloud& pointcloud) {
	try {
		input_file.read((char*)&pointcloud.height, sizeof(int32_t));
		input_file.read((char*)&pointcloud.width, sizeof(int32_t));
		input_file.read((char*)&pointcloud.point_step, sizeof(uint32_t));

		cudaMallocManaged(&pointcloud.data, pointcloud.height*pointcloud.width*pointcloud.point_step);
		input_file.read((char*)pointcloud.data, pointcloud.height*pointcloud.width*pointcloud.point_step);
    }  catch (std::ifstream::failure) {
		throw std::ios_base::failure("Error reading the next point cloud.");
    }
}

void points2image::cleanupTestcases(int count) {
	for (int i = 0; i < count; i++) {
		cudaFree(pointcloud[i].data);
		// avoid double free in framework code
		pointcloud[i].data = nullptr;
		// prevent free of result_buffer
		results[i].distance = nullptr;
		results[i].intensity = nullptr;
		results[i].min_height = nullptr;
		results[i].max_height = nullptr;
	}
	points2image_base::cleanupTestcases(count);
}
void points2image::init() {
	std::cout << "init\n";
	points2image_base::init();
	std::cout << "done\n" << std::endl;
}
void points2image::quit() {
	points2image_base::quit();
}
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
 * The Cuda kernel to execute.
 * Performs the transformation for a single point.
 * cp: pointer to cloud memory
 * msg_distance: image distances
 * msg_intensity: image intensities
 * msg_min_height: image minimum heights
 * width: cloud width
 * height: cloud height
 * point_step: point stride used for indexing cloud memory
 * w: image width
 * h: image height
 * invR: matrix to apply to cloud points
 * invT: translation to apply to cloud points
 * distCoeff: distance coefficients to apply to cloud points
 * cameraMat: camera intrinsic matrix
 * min_y: lower image extend bound
 * max_y: higher image extend bound
 */
__global__ void compute_point_from_pointcloud(
	const float*  __restrict__ cp,
	float*  volatile msg_distance,
	float* volatile msg_intensity,
	float * __restrict__ msg_min_height,
	int* volatile msg_access,
	int width, int height, int point_step, int w, int h,
	Mat33 invR, Mat13 invT, Vec5 distCoeff, Mat33 cameraMat,
	int* __restrict__ min_y, int* __restrict__ max_y) {

	// determine index in cloud memory
	int y = blockIdx.x;
	int x = blockIdx.y * THREADS + threadIdx.x;

	int px, py;
	float depth, intensity;
	bool isInside = (x < width);
	if (isInside) {

		float* fp = (float *)((uintptr_t)cp + (x + y*width) * point_step);

		intensity = fp[4];
		// first step of the transformation
		Mat13 point, point2;
		point2.data[0] = double(fp[0]);
		point2.data[1] = double(fp[1]);
		point2.data[2] = double(fp[2]);

		for (int row = 0; row < 3; row++) {
			point.data[row] = invT.data[row];
			for (int col = 0; col < 3; col++)
			point.data[row] += point2.data[col] * invR.data[row][col];
		}
		isInside = (point.data[2] > 2.5);
		if (isInside) {

			// second transformation step
			double tmpx = point.data[0]/point.data[2];
			double tmpy = point.data[1]/point.data[2];
			double r2 = tmpx * tmpx + tmpy * tmpy;
			double tmpdist = 1 + distCoeff.data[0] * r2
				+ distCoeff.data[1] * r2 * r2
				+ distCoeff.data[4] * r2 * r2 * r2;

			Point2d imagepoint;
			imagepoint.x = tmpx * tmpdist
				+ 2 * distCoeff.data[2] * tmpx * tmpy
				+ distCoeff.data[3] * (r2 + 2 * tmpx * tmpx);
			imagepoint.y = tmpy * tmpdist
				+ distCoeff.data[2] * (r2 + 2 * tmpy * tmpy)
				+ 2 * distCoeff.data[3] * tmpx * tmpy;
			// apply camera intrinsics to yield a point on the image
			imagepoint.x = cameraMat.data[0][0] * imagepoint.x + cameraMat.data[0][2];
			imagepoint.y = cameraMat.data[1][1] * imagepoint.y + cameraMat.data[1][2];
			px = int(imagepoint.x + 0.5);
			py = int(imagepoint.y + 0.5);

			isInside = (0 <= px && px < w && 0 <= py && py < h);
			// safe point characteristics in the image
			if (isInside) {
				int pid = py * w + px;
				depth = float(point.data[2] * 100.0);
				atomicCAS((int*)&msg_distance[pid], 0, __float_as_int(depth));
				atomicFloatMin(&msg_distance[pid], depth);
				//atomicAdd(&msg_access[pid], 1);
				msg_access[pid] = 0;
			}
		}
	}
	// synchronize required for deterministic intensity in the image
	__syncthreads();
	// note: we do not expect to have multiple devices, so __threadfence() should be enough here
	//__threadfence_system();
	__threadfence();
	if (isInside) {
		// since we used a threadfence before we should read the same value across all threads
		int pid = py * w + px;
		atomicAdd(&msg_access[pid], 1);
		float finalDistance = msg_distance[pid];
		// note: since we get issues with too high intensity values it seems like
		// too many threads satisfy the condition below
		// maybe they see different final distance values?
		// update intensity, height and image extends
		if (finalDistance == depth)
		{

			atomicFloatMax(&msg_intensity[pid], float(intensity));
			atomicMax(max_y, py);
			atomicMin(min_y, py);
			//atomicAdd(&msg_access[pid], 1);
		}
		msg_min_height[pid] = -1.25;
	}
	__syncthreads();

}

__global__ void computeCloudTransformation(
	const float* __restrict__ pointcloud,
	TransformPixel* __restrict__ pixels,
	int* pixelNo,
	const TransformationInfo* info) {

	__shared__ int b_pixelSort;
	__shared__ int b_pixelStart;
	__shared__ int b_pixelNo;

	if (threadIdx.x == 0) {
		b_pixelNo = 0;
	}
	__syncthreads();


	int iPoint = blockIdx.x*blockDim.x + threadIdx.x;
	int offset;
	TransformPixel pixel = { NULL };
	bool isInside = (iPoint < info->pointNo);
	if (isInside) {



		float* fp = (float *)((uintptr_t)pointcloud + iPoint*info->pointStep);
		Mat13 point, point2;
		point2.data[0] = double(fp[0]);
		point2.data[1] = double(fp[1]);
		point2.data[2] = double(fp[2]);

		for (int row = 0; row < 3; row++) {
			point.data[row] = info->invTranslation.data[row];
			for (int col = 0; col < 3; col++)
			point.data[row] += point2.data[col]*info->invRotation.data[row][col];
		}
		isInside = (point.data[2] > 2.5);
		if (isInside) {

			// second transformation step
			double tmpx = point.data[0]/point.data[2];
			double tmpy = point.data[1]/point.data[2];
			double r2 = tmpx * tmpx + tmpy * tmpy;
			double tmpdist = 1 + info->distCoeff.data[0] * r2
				+ info->distCoeff.data[1] * r2 * r2
				+ info->distCoeff.data[4] * r2 * r2 * r2;

			Point2d imagepoint;
			imagepoint.x = tmpx * tmpdist
				+ 2 * info->distCoeff.data[2] * tmpx * tmpy
				+ info->distCoeff.data[3] * (r2 + 2 * tmpx * tmpx);
			imagepoint.y = tmpy * tmpdist
				+ info->distCoeff.data[2] * (r2 + 2 * tmpy * tmpy)
				+ 2 * info->distCoeff.data[3] * tmpx * tmpy;
			// apply camera intrinsics to yield a point on the image
			imagepoint.x = info->cameraMat.data[0][0] * imagepoint.x + info->cameraMat.data[0][2];
			imagepoint.y = info->cameraMat.data[1][1] * imagepoint.y + info->cameraMat.data[1][2];
			pixel.posX = int(imagepoint.x + 0.5);
			pixel.posY = int(imagepoint.y + 0.5);

			isInside = (0 <= pixel.posX && pixel.posX < info->imgWidth &&
						0 <= pixel.posY && pixel.posY < info->imgHeight);
			// safe point characteristics in the image
			if (isInside) {
				pixel.depth = float(point.data[2]*100.0);
				pixel.intensity = fp[4];
				int offset = atomicAdd(pixelNo, 1);
				pixels[offset] = pixel;
			}
		}
	}
// 	if (isInside) {
//
// 		float* fp = (float*)((uintptr_t)pointcloud + iPoint*info.pointStep);
//
// 		// first step of the transformation
// 		Mat13 point;
//
//
// 		for (int row = 0; row < 3; row++) {
// 			point.data[row] = info.invTranslation.data[row];
// 			for (int col = 0; col < 3; col++)
// 			point.data[row] += double(fp[col])*info.invRotation.data[row][col];
// 		}
// 		isInside = (point.data[2] > 2.5);
// 		if (isInside) {
//
// 			// second transformation step
// 			double tmpx = point.data[0]/point.data[2];
// 			double tmpy = point.data[1]/point.data[2];
// 			double r2 = tmpx * tmpx + tmpy * tmpy;
// 			double tmpdist = 1 + info.distCoeff.data[0] * r2
// 				+ info.distCoeff.data[1] * r2 * r2
// 				+ info.distCoeff.data[4] * r2 * r2 * r2;
//
// 			Point2d imagepoint;
// 			imagepoint.x = tmpx * tmpdist
// 				+ 2 * info.distCoeff.data[2] * tmpx * tmpy
// 				+ info.distCoeff.data[3] * (r2 + 2 * tmpx * tmpx);
// 			imagepoint.y = tmpy * tmpdist
// 				+ info.distCoeff.data[2] * (r2 + 2 * tmpy * tmpy)
// 				+ 2 * info.distCoeff.data[3] * tmpx * tmpy;
// 			// apply camera intrinsics to yield a point on the image
// 			imagepoint.x = info.camScale.data[0]*imagepoint.x + info.camOffset.data[0];
// 			imagepoint.y = info.camScale.data[1]*imagepoint.y + info.camOffset.data[1];
// 			pixel.posX = int(imagepoint.x + 0.5);
// 			pixel.posY = int(imagepoint.y + 0.5);
//
// 			isInside = (0 <= pixel.posX && pixel.posX < info.imgWidth &&
// 				0 <= pixel.posY && pixel.posY < info.imgHeight);
// 			// safe point characteristics in the image
// 			if (isInside) {
// 				pixel.depth = float(point.data[2]*100.0);
// 				pixel.intensity = fp[4];
// 				offset = atomicAdd(&b_pixelNo, 1);
// 			}
// 		}
// 	}

// 	__syncthreads();
// 	int pixNo = b_pixelNo;
// 	//int pixNo = __syncthreads_count(isInside);
// 	if (pixNo > 0) {
// 		if (threadIdx.x == 0) {
// 			b_pixelSort = 0;
// 			b_pixelStart = atomicAdd(pixelNo, pixNo);
// 		}
// 		__syncthreads();
// 		if (isInside) {
// 			//int iPixel = b_pixelStart + atomicAdd(&b_pixelSort, 1);
// 			int iPixel = b_pixelStart + offset;
// 			pixels[iPixel] = pixel;
//
// 		}
// 	}
}



/**
 * This code is extracted from Autoware, file:
 * ~/Autoware/ros/src/sensing/fusion/packages/points2image/lib/points_image/points_image.cpp
 * It uses the test data that has been read before and applies the linked algorithm.
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
	ImageSize& imageSize)
{
	// initialize the resulting image data structure
// 	int w = imageSize.width;
// 	int h = imageSize.height;
// 	PointsImage msg;
// 	msg.max_y = -1;
// 	msg.min_y = h;
// 	msg.image_height = imageSize.height;
// 	msg.image_width = imageSize.width;
// 	msg.intensity = result_buffer;
// 	msg.distance = msg.intensity + h*w;
// 	msg.min_height = msg.distance + h*w;
// 	msg.max_height = msg.min_height + h*w;
// 	int* msg_access = (int*)(msg.max_height + h*w);
// 	std::memset(msg.intensity, 0, sizeof(float)*w*h);
// 	//std::fill(msg.intensity, msg.intensity + w*h, 256.0f);
// 	std::memset(msg.distance, 0, sizeof(float)*w*h);
// 	std::memset(msg.min_height, 0, sizeof(float)*w*h);
// 	std::memset(msg.max_height, 0, sizeof(float)*w*h);
// 	std::memset(msg_access, 0, sizeof(int)*w*h);
//
// 	// preprocess the given matrices
// 	Mat33 invR;
// 	Mat13 invT;
// 	// transposed 3x3 camera extrinsic matrix
// 	for (int row = 0; row < 3; row++)
// 		for (int col = 0; col < 3; col++)
// 			invR.data[row][col] = cameraExtrinsicMat.data[col][row];
// 	// translation vector: (transposed camera extrinsic matrix)*(fourth column of camera extrinsic matrix)
// 	for (int row = 0; row < 3; row++) {
// 		invT.data[row] = 0.0;
// 		for (int col = 0; col < 3; col++)
// 			invT.data[row] -= invR.data[row][col] * cameraExtrinsicMat.data[col][3];
// 	}
// 	// allocate memory for additional information used by the kernel
// 	int *cuda_min_y, *cuda_max_y;
// 	cudaMalloc(&cuda_min_y, sizeof(int));
// 	cudaMalloc(&cuda_max_y, sizeof(int));
// 	cudaMemcpy(cuda_min_y, &msg.min_y, sizeof(int), cudaMemcpyHostToDevice);
// 	cudaMemcpy(cuda_max_y, &msg.max_y, sizeof(int), cudaMemcpyHostToDevice);
//
// 	// call the kernel with enough threads to process the cloud in a single call
// 	dim3 threaddim(THREADS);
// 	dim3 blockdim(pointcloud.height, (pointcloud.width+THREADS-1)/THREADS);
// 	compute_point_from_pointcloud<<<blockdim, threaddim>>>(pointcloud.data, msg.distance,
// 								msg.intensity, msg.min_height, msg_access,
// 								pointcloud.width, pointcloud.height, pointcloud.point_step,
// 								w, h,
// 								invR, invT, distCoeff, cameraMat,
// 								cuda_min_y, cuda_max_y);
// 	// wait for the result and read image extends
// 	cudaDeviceSynchronize();
// 	cudaMemcpy(&msg.min_y, cuda_min_y, sizeof(int),
// 		cudaMemcpyDeviceToHost);
// 	cudaMemcpy(&msg.max_y, cuda_max_y, sizeof(int),
// 		cudaMemcpyDeviceToHost);
// 	cudaFree(cuda_max_y);
// 	cudaFree(cuda_min_y);
//
// 	for (int y = 0; y < h; y++) {
// 		for (int x = 0; x < w; x++) {
// 			int pid = w*y + x;
// 			if (msg.distance[pid] == 0.0f) {
//  				msg.intensity[pid] = 0.0f;
// 			}
// 		}
// 	}
	int w = imageSize.width;
	int h = imageSize.height;
	int imagePixelNo = imageSize.width*imageSize.height;
	int pointNo = pointcloud.width*pointcloud.height;
	PointsImage msg;
	msg.max_y = -1;
	msg.min_y = h;
	msg.image_height = imageSize.height;
	msg.image_width = imageSize.width;
	msg.intensity = result_buffer;
	msg.distance = msg.intensity + imagePixelNo;
	msg.min_height = msg.distance + imagePixelNo;
	msg.max_height = msg.min_height + imagePixelNo;
	int* msg_access = (int*)(msg.max_height + imagePixelNo);
	std::memset(msg.intensity, 0, sizeof(float)*imagePixelNo);
	//std::fill(msg.intensity, msg.intensity + w*h, 256.0f);
	std::memset(msg.distance, 0, sizeof(float)*imagePixelNo);
	std::memset(msg.min_height, 0, sizeof(float)*imagePixelNo);
	std::memset(msg.max_height, 0, sizeof(float)*imagePixelNo);
	std::memset(msg_access, 0, sizeof(int)*imagePixelNo);

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
	TransformationInfo* computeInfo;
	cudaMallocManaged(&computeInfo, sizeof(TransformationInfo));

	*computeInfo = (TransformationInfo){
		cameraMat,
		invR,
		distCoeff,
		invT,
		//{ cameraMat.data[0][2], cameraMat.data[1][2] },
		//{ cameraMat.data[0][0], cameraMat.data[1][1] },
		pointNo,
		pointcloud.point_step,
		imageSize.width, imageSize.height
	};
	dim3 computeThreadDim(THREADS);
	dim3 computeBlockDim((pointNo + THREADS - 1)/THREADS);
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
	cudaFree(computeInfo);


	PointsImage msg2;
	msg2.intensity = new float[w*h];
	std::memset(msg2.intensity, 0, sizeof(float)*w*h);
	msg2.distance = new float[w*h];
	std::memset(msg2.distance, 0, sizeof(float)*w*h);

	int* msg2_access = new int[w*h];
	std::memset(msg2_access, 0, sizeof(int)*w*h);
	msg2.max_y = -1;
	msg2.min_y = h;
	msg2.image_height = imageSize.height;
	msg2.image_width = imageSize.width;
	int pixelsInside = 0;
	// prepare cloud data pointer to read the data correctly
	{
		uintptr_t cp = (uintptr_t)pointcloud.data;

		// preprocess the given matrices
		// transposed 3x3 camera extrinsic matrix
		Mat33 invR;
		for (int row = 0; row < 3; row++)
			for (int col = 0; col < 3; col++)
				invR.data[row][col] = cameraExtrinsicMat.data[col][row];
		// translation vector: (transposed camera extrinsic matrix)*(fourth column of camera extrinsic matrix)
		Mat13 invT;
		for (int row = 0; row < 3; row++) {
			invT.data[row] = 0.0;
			for (int col = 0; col < 3; col++)
				invT.data[row] -= invR.data[row][col] * cameraExtrinsicMat.data[col][3];
		}
		// apply the algorithm for each point in the cloud
		for (uint32_t y = 0; y < pointcloud.height; ++y) {
			for (uint32_t x = 0; x < pointcloud.width; ++x) {
				// the start of the current point in the cloud to process
				float* fp = (float *)(cp + (x + y*pointcloud.width) * pointcloud.point_step);
				double intensity = fp[4];

				Mat13 point, point2;
				point2.data[0] = double(fp[0]);
				point2.data[1] = double(fp[1]);
				point2.data[2] = double(fp[2]);

				// start the the predetermined translation
				for (int row = 0; row < 3; row++) {
					point.data[row] = invT.data[row];
				// add the transformed cloud point
				for (int col = 0; col < 3; col++)
					point.data[row] += point2.data[col] * invR.data[row][col];
				}
				// discard points with small depth values
				if (point.data[2] <= 2.5) {
					continue;
				}
				// perform perspective division
				double tmpx = point.data[0]/point.data[2];
				double tmpy = point.data[1]/point.data[2];
				// apply the distance coefficients
				double r2 = tmpx * tmpx + tmpy * tmpy;
				double tmpdist = 1 + distCoeff.data[0] * r2
				+ distCoeff.data[1] * r2 * r2
				+ distCoeff.data[4] * r2 * r2 * r2;

				Point2d imagepoint;
				imagepoint.x = tmpx * tmpdist
				+ 2 * distCoeff.data[2] * tmpx * tmpy
				+ distCoeff.data[3] * (r2 + 2 * tmpx * tmpx);
				imagepoint.y = tmpy * tmpdist
				+ distCoeff.data[2] * (r2 + 2 * tmpy * tmpy)
				+ 2 * distCoeff.data[3] * tmpx * tmpy;

				// apply the camera matrix (camera intrinsics) and end up with a two dimensional point
				imagepoint.x = cameraMat.data[0][0] * imagepoint.x + cameraMat.data[0][2];
				imagepoint.y = cameraMat.data[1][1] * imagepoint.y + cameraMat.data[1][2];
				int px = int(imagepoint.x + 0.5);
				int py = int(imagepoint.y + 0.5);

				// continue with points that landed inside image bounds
				if(0 <= px && px < w && 0 <= py && py < h)
				{
					pixelsInside += 1;
					// target pixel index linearization
					int pid = py * w + px;
					msg2_access[pid] += 1;
					// replace unset pixels as well as pixels with a higher distance value
					if(msg2.distance[pid] == 0.0 ||
						msg2.distance[pid] >= float(point.data[2] * 100.0))
					{
						// make the result always deterministic and independent from the point order
						// in case two points get the same distance, take the one with higher intensity
						if (((msg2.distance[pid] == float(point.data[2] * 100.0)) &&  msg2.intensity[pid] < float(intensity)) ||
							(msg2.distance[pid] > float(point.data[2] * 100.0)) ||
							msg2.distance[pid] == 0)
						{
							if (msg2.distance[pid] == float(point.data[2]*100.0)) {
								//msg2_access[pid] += 1;
							} else {
								//msg2_access[pid] = 1;
							}
							msg2.intensity[pid] = float(intensity);
						}
						msg2.distance[pid] = float(point.data[2] * 100.0);
					}
				}
			}
		}
	}
	int diff = 0;
	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			int pid = y*w + x;
			if (msg_access[pid] != msg2_access[pid]) {
				//std::cout << "[" << x << " " << y << "] accessed ";
				//std::cout << msg_access[pid] << " times instead of " << msg2_access[pid] << std::endl;
			}
			if (msg2_access[pid] > 2) {
				std::cout << "[" << x << " " << y << "] accessed ";
				std::cout << msg2_access[pid] << " times" << std::endl;
			}
		}
	}
	//std::cout << "diff " << diff << std::endl;
	delete[] msg2.distance;
	delete[] msg2.intensity;
	delete[] msg2_access;

	return msg;
}


// void points2image::check_next_outputs(int count)
// {
// 	PointsImage reference;
// 	// parse the next reference image
// 	// and compare it to the data generated by the algorithm
// 	for (int i = 0; i < count; i++)
// 	{
// 		std::ostringstream sError;
// 		int caseErrorNo = 0;
// 		try {
// 			parsePointsImage(output_file, reference);
// #ifdef EPHOS_TESTDATA_GEN
// 			writePointsImage(datagen_file, &results[i]);
// #endif
// 		} catch (std::ios_base::failure& e) {
// 			std::cerr << e.what() << std::endl;
// 			exit(-3);
// 		}
// 		// detect image size deviation
// 		if ((results[i].image_height != reference.image_height)
// 			|| (results[i].image_width != reference.image_width))
// 		{
// 			error_so_far = true;
// 			caseErrorNo += 1;
// 			sError << " deviating image size: [" << results[i].image_width << " ";
// 			sError << results[i].image_height << "] should be [";
// 			sError << reference.image_width << " " << reference.image_height << "]" << std::endl;
// 		}
// 		// detect image extend deviation
// 		if ((results[i].min_y != reference.min_y)
// 			|| (results[i].max_y != reference.max_y))
// 		{
// 			error_so_far = true;
// 			caseErrorNo += 1;
// 			sError << " deviating vertical intervall: [" << results[i].min_y << " ";
// 			sError << results[i].max_y << "] should be [";
// 			sError << reference.min_y << " " << reference.max_y << "]" << std::endl;
// 		}
// 		// compare all pixels
// 		int pos = 0;
// 		for (int h = 0; h < reference.image_height; h++)
// 			for (int w = 0; w < reference.image_width; w++)
// 			{
// 				// test for intensity
// 				float delta = std::fabs(reference.intensity[pos] - results[i].intensity[pos]);
// 				if (delta > max_delta) {
// 					max_delta = delta;
// 				}
// 				if (delta > MAX_EPS) {
// 					sError << " with distance " << reference.distance[pos];
// 					sError << " at [" << w << " " << h << "]: Intensity " << results[i].intensity[pos];
// 					sError << " should be " << reference.intensity[pos] << std::endl;
// 					caseErrorNo += 1;
// 				}
// 				// test for distance
// 				delta = std::fabs(reference.distance[pos] - results[i].distance[pos]);
// 				if (delta > max_delta) {
// 					max_delta = delta;
// 				}
// 				if (delta > MAX_EPS) {
// 					sError << " at [" << w << " " << h << "]: Distance " << results[i].distance[pos];
// 					sError << " should be " << reference.distance[pos] << std::endl;
// 					caseErrorNo += 1;
// 				}
// 				// test for min height
// 				delta = std::fabs(reference.min_height[pos] - results[i].min_height[pos]);
// 				if (delta > max_delta) {
// 					max_delta = delta;
// 				}
// 				if (delta > MAX_EPS) {
// 					sError << " at [" << w << " " << h << "]: Min height " << results[i].min_height[pos];
// 					sError << " should be " << reference.min_height[pos] << std::endl;
// 					caseErrorNo += 1;
// 				}
// 				// test for max height
// 				delta = std::fabs(reference.max_height[pos] - results[i].max_height[pos]);
// 				if (delta > max_delta) {
// 					max_delta = delta;
// 				}
// 				if (delta > MAX_EPS) {
// 					sError << " at [" << w << " " << h << "]: Max height " << results[i].max_height[pos];
// 					sError << " should be " << reference.max_height[pos] << std::endl;
// 					caseErrorNo += 1;
// 				}
// 				pos += 1;
// 			}
// 		if (caseErrorNo > 0) {
// 			std::cerr << "Errors for test case " << read_testcases - count + i;
// 			std::cerr << " (" << caseErrorNo << "):" << std::endl;
// 			std::cerr << sError.str() << std::endl;
// 		}
// 		// free the memory allocated by the reference image read above
// 		delete[] reference.intensity;
// 		delete[] reference.distance;
// 		delete[] reference.min_height;
// 		delete[] reference.max_height;
// 		// no need to delete from results
// 		// arrays in results are freed on program exit
// 		// but a free operation is required on the previously allocated input data
// 		cudaFree(pointcloud[i].data);
// 	}
// 	results.clear();
// 	cameraExtrinsicMat.clear();
// 	cameraMat.clear();
// 	distCoeff.clear();
// 	imageSize.clear();
// 	pointcloud.clear();
// }

// set the external kernel instance used in main()
points2image a;
benchmark& myKernel = a;
