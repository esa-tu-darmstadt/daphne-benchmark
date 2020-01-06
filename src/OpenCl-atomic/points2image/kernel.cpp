/**
 * Author:  Florian Stock, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2018
 * Author:  Leonardo Solis, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2018
 * Author:  Thilo Gabel, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2019
 * License: Apache 2.0 (see attachached File)
 */
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdint>
#include <cstring>
#include <string>

#include "kernel.h"
#include "benchmark.h"
#include "datatypes.h"
#include "ocl/device/ocl_kernel.h"
#include "ocl/host/ocl_header.h"

void points2image::init() {
	std::cout << "init\n";
	
	// open testcase and reference data streams
	input_file.exceptions ( std::ifstream::failbit | std::ifstream::badbit );
	output_file.exceptions ( std::ifstream::failbit | std::ifstream::badbit );
	try {
		input_file.open("../../../data/p2i_input.dat", std::ios::binary);
	} catch (std::ifstream::failure) {
		std::cerr << "Error opening the input data file" << std::endl;
		exit(-2);
	}
	try {
		output_file.open("../../../data/p2i_output.dat", std::ios::binary);
	} catch (std::ifstream::failure) {
		std::cerr << "Error opening the output data file" << std::endl;
		exit(-2);
	}
	try {
	// consume the total number of testcases
		testcases = read_number_testcases(input_file);
	} catch (std::ios_base::failure& e) {
		std::cerr << e.what() << std::endl;
		exit(-3);
	}
	
	// prepare the first iteration
	error_so_far = false;
	max_delta = 0.0;
	pointcloud2 = nullptr;
	cameraExtrinsicMat = nullptr;
	cameraMat = nullptr;
	distCoeff = nullptr;
	imageSize = nullptr;
	results = nullptr;

	std::cout << "done\n" << std::endl;
}

void points2image::run(int p) {
	// do not measure setup time
	pause_func();
	OCL_Struct OCL_objs;
	try {
	    std::vector<std::vector<std::string>> extensions = { {"cl_khr_fp64", "cl_amd_fp64" } };
	    OCL_objs = OCL_Tools::find_compute_platform(EPHOS_PLATFORM_HINT_S, EPHOS_DEVICE_HINT_S,
			EPHOS_DEVICE_TYPE_S, extensions);
	} catch (std::logic_error& e) {
	    std::cerr << "OpenCL setup failed. " << e.what() << std::endl;
	}
	{ // display used device
		std::vector<char> nameBuffer(256);
		size_t nameLength = 0;
		clGetDeviceInfo(OCL_objs.device, CL_DEVICE_NAME, nameBuffer.size(), nameBuffer.data(), &nameLength);
		std::cout << "EPHoS OpenCL device: " << std::string(nameBuffer.data(), nameLength) << std::endl;
	}

	std::vector<cl_kernel> kernels;
	cl_program points2image_program;
	try {
		std::vector<std::string> kernelNames({
			"pointcloud2_to_image"
		});
		std::string sOptions =
#ifdef EPHOS_KERNEL_ATOMICS
			"-DEPHOS_KERNEL_ATOMICS"
#else
			""
#endif
			;
		std::string sSource(points2image_ocl_krnl);
		points2image_program = OCL_Tools::build_program(OCL_objs, sSource, sOptions,
			kernelNames, kernels);
	} catch (std::logic_error& e) {
		std::cerr << e.what() << std::endl;
		exit(EXIT_FAILURE);
	}
	cl_kernel points2imageKernel = kernels[0];
	cl_int err = CL_SUCCESS;
	// process all testcases
	while (read_testcases < testcases)
	{
		// read the testcase data, then start the computation
		int count = read_next_testcases(p);
		unpause_func();
		// Set kernel parameters & launch NDRange kernel
		for (int i = 0; i < count; i++)
		{
			// Prepare inputs buffers
			size_t pointNo = pointcloud2[i].height * pointcloud2[i].width * pointcloud2[i].point_step;
			size_t cloudSize = pointNo * sizeof(float);
			cl_mem cloudBuffer = clCreateBuffer(OCL_objs.context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY, cloudSize, NULL, &err);
			// write cloud input to buffer
			err = clEnqueueWriteBuffer(OCL_objs.cmdqueue, cloudBuffer, CL_FALSE, 0, cloudSize, pointcloud2[i].data, 0, nullptr, nullptr);
			// Prepare outputs buffers
			size_t imagePixelNo = imageSize[i].height*imageSize[i].width;
			// Allocate space in host to store results comming from GPU
			// These will be freed in read_next_testcases()
			results[i].intensity  = new float[imagePixelNo];
			std::memset(results[i].intensity, 0, sizeof(float)*imagePixelNo);
			results[i].distance   = new float[imagePixelNo];
			std::memset(results[i].distance, 0, sizeof(float)*imagePixelNo);
			results[i].min_height = new float[imagePixelNo];
			std::memset(results[i].min_height, 0, sizeof(float)*imagePixelNo);
			results[i].max_height = new float[imagePixelNo];
			std::memset(results[i].max_height, 0, sizeof(float)*imagePixelNo);
			// Creating zero-copy buffers for pids data
			cl_mem pixelIdBuffer = clCreateBuffer(OCL_objs.context, CL_MEM_WRITE_ONLY, pointcloud2[i].width * sizeof(int),   nullptr, &err);
			cl_mem depthBuffer = clCreateBuffer(OCL_objs.context, CL_MEM_WRITE_ONLY, pointcloud2[i].width * sizeof(float), nullptr, &err);
			cl_mem intensityBuffer = clCreateBuffer(OCL_objs.context, CL_MEM_WRITE_ONLY, pointcloud2[i].width * sizeof(float), nullptr, &err);
			cl_mem counterBuffer = clCreateBuffer(OCL_objs.context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(int), nullptr, &err);
			// Set kernel parameters
			err = clSetKernelArg (points2imageKernel, 0, sizeof(int),       &pointcloud2[i].height);
			err = clSetKernelArg (points2imageKernel, 1, sizeof(int),       &pointcloud2[i].width);
			err = clSetKernelArg (points2imageKernel, 2, sizeof(int),       &pointcloud2[i].point_step);
			err = clSetKernelArg (points2imageKernel, 3, sizeof(cl_mem),    &cloudBuffer);
			// prepare matrices
			Mat44 tmpCameraExtrinsic;
			Mat33 tmpCameraMat;
			Vec5  tmpDistCoeff;

			for (uint p=0; p<4; p++){
				for (uint q=0; q<4; q++) {
					tmpCameraExtrinsic.data[p][q] = cameraExtrinsicMat[i].data[p][q];
				}
			}
			for (uint p=0; p<3; p++){
				for (uint q=0; q<3; q++) {
					tmpCameraMat.data[p][q] = cameraMat[i].data[p][q];
				}
			}
			for (uint p=0; p<5; p++){
				tmpDistCoeff.data[p] = distCoeff[i].data[p];
			}

			err = clSetKernelArg (points2imageKernel, 4,  sizeof(Mat44),  &tmpCameraExtrinsic);
			err = clSetKernelArg (points2imageKernel, 5,  sizeof(Mat33),  &tmpCameraMat);
			err = clSetKernelArg (points2imageKernel, 6,  sizeof(Vec5),   &tmpDistCoeff);

			err = clSetKernelArg (points2imageKernel, 7,  sizeof(ImageSize), &imageSize[i]);
			err = clSetKernelArg (points2imageKernel, 8,  sizeof(cl_mem), &pixelIdBuffer);
			err = clSetKernelArg (points2imageKernel, 9, sizeof(cl_mem), &depthBuffer);
			err = clSetKernelArg (points2imageKernel, 10, sizeof(cl_mem), &intensityBuffer);
			err = clSetKernelArg (points2imageKernel, 11, sizeof(cl_mem), &counterBuffer);
			err = clSetKernelArg(points2imageKernel, 12, sizeof(int), nullptr);
			err = clSetKernelArg(points2imageKernel, 13, sizeof(int), nullptr);
			// initializing arriving point number
			int zero = 0;
			err = clEnqueueWriteBuffer(OCL_objs.cmdqueue, counterBuffer, CL_FALSE,
				0, sizeof(int), &zero, 0, nullptr, nullptr);
			// Launch kernel on device
			size_t localRange = NUMWORKITEMS_PER_WORKGROUP;
			size_t globalRange = (pointcloud2[i].width/localRange + 1)*localRange;
			err = clEnqueueNDRangeKernel(OCL_objs.cmdqueue, points2imageKernel, 1,
				nullptr,  &globalRange, &localRange, 0, nullptr, nullptr);

			int arrivingPointNo;
			err = clEnqueueReadBuffer(OCL_objs.cmdqueue, counterBuffer, CL_TRUE,
				0, sizeof(int), &arrivingPointNo, 0, nullptr, nullptr);
			// move results to host memory
			int* pixelIds = (int*) clEnqueueMapBuffer(OCL_objs.cmdqueue, pixelIdBuffer, 
				CL_TRUE, CL_MAP_READ, 0, sizeof(int)*arrivingPointNo, 0, 0, nullptr, &err);
			float* pointDepth = (float*) clEnqueueMapBuffer(OCL_objs.cmdqueue, depthBuffer,
				CL_TRUE, CL_MAP_READ, 0, sizeof(float)*arrivingPointNo, 0, 0, nullptr, &err);
			float* pointIntensity = (float*) clEnqueueMapBuffer(OCL_objs.cmdqueue, intensityBuffer,
				CL_TRUE, CL_MAP_READ, 0, sizeof(float)*arrivingPointNo, 0, 0, nullptr, &err);
			const int h          = imageSize[i].height;
			const int pc2_height = pointcloud2[i].height;
			const int pc2_width  = pointcloud2[i].width;
			const int pc2_pstep  = pointcloud2[i].point_step;
			results[i].max_y        = -1;
			results[i].min_y        = h;
			results[i].image_height = imageSize[i].height;
			results[i].image_width  = imageSize[i].width;
			uintptr_t cp = (uintptr_t)pointcloud2[i].data;
			// transfer the transformation results into the image
			for (unsigned int x = 0; x < arrivingPointNo; x++) {
				int pid = pixelIds[x];
				float tmpDepth = pointDepth[x] * 100;
				float tmpDistance = results[i].distance[pid];

				bool cond1 = (tmpDistance == 0.0f);
				bool cond2 = (tmpDistance >= tmpDepth);
				if( cond1 || cond2 ) {
					bool cond3 = (tmpDistance == tmpDepth);
					bool cond4 = (results[i].intensity[pid] <  pointIntensity[x]);
					bool cond5 = (tmpDistance >  tmpDepth);
					bool cond6 = (tmpDistance == 0);

					if ((cond3 && cond4) || cond5 || cond6) {
						results[i].intensity[pid] = pointIntensity[x];
					}
					results[i].distance[pid]  = float(tmpDepth);
					int pixelY = pid/imageSize[i].width;
					if (results[i].max_y < pixelY) {
						results[i].max_y = pixelY;
					}
					if (results[i].min_y > pixelY) {
						results[i].min_y = pixelY;
					}
				}
				results[i].min_height[pid] = -1.25f;
				results[i].max_height[pid] = 0.0f;
			}
			// cleanup
			clEnqueueUnmapMemObject(OCL_objs.cmdqueue, pixelIdBuffer, pixelIds, 0, nullptr, nullptr);
			clEnqueueUnmapMemObject(OCL_objs.cmdqueue, depthBuffer, pointDepth, 0, nullptr, nullptr);
			clEnqueueUnmapMemObject(OCL_objs.cmdqueue, intensityBuffer, pointIntensity, 0, nullptr, nullptr);

			clReleaseMemObject(cloudBuffer);
			clReleaseMemObject(pixelIdBuffer);
			clReleaseMemObject(depthBuffer);
			clReleaseMemObject(intensityBuffer);
			clReleaseMemObject(counterBuffer);
		}
		pause_func();
		check_next_outputs(count);
	}
	// cleanup
	err = clReleaseKernel(points2imageKernel);
	err = clReleaseProgram(points2image_program);
	err = clReleaseCommandQueue(OCL_objs.cmdqueue);
	err = clReleaseContext(OCL_objs.context);
}


