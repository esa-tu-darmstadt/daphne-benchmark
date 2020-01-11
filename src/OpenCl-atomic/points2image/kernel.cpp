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
#ifdef EPHOS_TESTCASE_LIMIT
	testcases = std::min(testcases, (uint32_t)EPHOS_TESTCASE_LIMIT);
#endif
	std::cout << "Executing for " << testcases << " test cases" << std::endl;
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

			// Prepare outputs data structures
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
			results[i].max_y        = -1;
			results[i].min_y        = imageSize[i].height;
			results[i].image_height = imageSize[i].height;
			results[i].image_width  = imageSize[i].width;
			// prepare inputs buffers
			size_t pointNo = pointcloud2[i].height*pointcloud2[i].width;
			size_t cloudSize = pointNo*pointcloud2[i].point_step*sizeof(float);
			cl_mem cloudBuffer = clCreateBuffer(OCL_objs.context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY, cloudSize, NULL, &err);
			// write cloud input to buffer
			err = clEnqueueWriteBuffer(OCL_objs.cmdqueue, cloudBuffer, CL_FALSE, 0, cloudSize, pointcloud2[i].data, 0, nullptr, nullptr);
			// prepare output buffers
			cl_mem arrivingPixelBuffer = clCreateBuffer(OCL_objs.context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, pointNo*sizeof(PixelData), nullptr, &err);
			cl_mem counterBuffer = clCreateBuffer(OCL_objs.context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(int), nullptr, &err);
			// transformation info for kernel
			double (*c)[4] = &cameraExtrinsicMat[i].data[0];
			TransformInfo transformInfo {
				{ 0.0, 0.0, 0.0, // initial rotation
				  0.0, 0.0, 0.0,
				  0.0, 0.0, 0.0 },
				{ 0.0, 0.0, 0.0 }, // initial translation
				{ cameraMat[i].data[0][0], cameraMat[i].data[1][1] }, // camera scale
				{ cameraMat[i].data[0][2] + 0.5, cameraMat[i].data[1][2] + 0.5}, // camera offset
				{ distCoeff[i].data[0], // distortion coefficients
				  distCoeff[i].data[1],
				  distCoeff[i].data[2],
				  distCoeff[i].data[3],
				  distCoeff[i].data[4] },
				{ imageSize[i].width, imageSize[i].height }, // image size
				pointcloud2[i].width*pointcloud2[i].height, // cloud point number
				(int)(pointcloud2[i].point_step/sizeof(float)) // cloud point step
			};
			// calculate initial rotation and translation
			for (int row = 0; row < 3; row++) {
				for (int col = 0; col < 3; col++) {
					transformInfo.initRotation[row][col] = c[col][row];
					transformInfo.initTranslation[row] -= transformInfo.initRotation[row][col]*c[col][3];
				}
			}
			// set kernel parameters
			err = clSetKernelArg (points2imageKernel, 0, sizeof(TransformInfo),       &transformInfo);
			err = clSetKernelArg (points2imageKernel, 1, sizeof(cl_mem),    &cloudBuffer);
			err = clSetKernelArg(points2imageKernel, 2, sizeof(cl_mem), &arrivingPixelBuffer);
			err = clSetKernelArg (points2imageKernel, 3, sizeof(cl_mem), &counterBuffer);
			err = clSetKernelArg(points2imageKernel, 4, sizeof(int), nullptr);
			err = clSetKernelArg(points2imageKernel, 5, sizeof(int), nullptr);
			// initializing arriving point number
			int zero = 0;
			err = clEnqueueWriteBuffer(OCL_objs.cmdqueue, counterBuffer, CL_FALSE,
				0, sizeof(int), &zero, 0, nullptr, nullptr);
			// launch kernel on device
			size_t localRange = NUMWORKITEMS_PER_WORKGROUP;
			size_t globalRange = (pointNo/localRange + 1)*localRange;
			err = clEnqueueNDRangeKernel(OCL_objs.cmdqueue, points2imageKernel, 1,
				nullptr,  &globalRange, &localRange, 0, nullptr, nullptr);

#ifdef EPHOS_KERNEL_ATOMICS
			int arrivingPixelNo;
			// read arriving pixel number from buffer
			err = clEnqueueReadBuffer(OCL_objs.cmdqueue, counterBuffer, CL_TRUE,
				0, sizeof(int), &arrivingPixelNo, 0, nullptr, nullptr);
#else // !EPHOS_KERNEL_ATOMICS
			int arrivingPixelNo = pointNo;
#endif // !EPHOS_KERNEL_ATOMICS
			// process arriving pixels
			PixelData* arrivingPixelStorage = (PixelData*)clEnqueueMapBuffer(OCL_objs.cmdqueue,
				arrivingPixelBuffer, CL_TRUE, CL_MAP_READ, 0, sizeof(PixelData)*arrivingPixelNo, 0, 0, nullptr, &err);

			for (int j = 0; j < arrivingPixelNo; j++) {
				if (arrivingPixelStorage[j].position[0] > -1) {
					int iPixel = arrivingPixelStorage[j].position[1]*imageSize[i].width + arrivingPixelStorage[j].position[0];
					float currentDepth = results[i].distance[iPixel];
					float nextDepth = arrivingPixelStorage[j].depth*100.0f;

					if ((currentDepth == 0.0f) || (nextDepth <= currentDepth)) {
						float currentIntensity = results[i].intensity[iPixel];
						float nextIntensity = arrivingPixelStorage[j].intensity;
						// update intensity
						if ((currentDepth == nextDepth && nextIntensity > currentIntensity) ||
							(nextDepth < currentDepth) ||
							(currentDepth == 0)) {

							results[i].intensity[iPixel] = nextIntensity;
						}
						// update depth
						results[i].distance[iPixel] = nextDepth;
						// update height
						results[i].min_height[iPixel] = -1.25f;
						results[i].max_height[iPixel] = 0.0f;
						// update extends
						if (arrivingPixelStorage[j].position[1] > results[i].max_y) {
							results[i].max_y = arrivingPixelStorage[j].position[1];
						}
						if (arrivingPixelStorage[j].position[1] < results[i].min_y) {
							results[i].min_y = arrivingPixelStorage[j].position[1];
						}
					}
				}
			}
			// case cleanup
			clReleaseMemObject(arrivingPixelBuffer);
			clReleaseMemObject(cloudBuffer);
			clReleaseMemObject(counterBuffer);
		}
		pause_func();
		check_next_outputs(count);
	}
	// benchmark cleanup
	err = clReleaseKernel(points2imageKernel);
	err = clReleaseProgram(points2image_program);
	err = clReleaseCommandQueue(OCL_objs.cmdqueue);
	err = clReleaseContext(OCL_objs.context);
}


