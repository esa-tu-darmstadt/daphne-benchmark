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

#include "points2image.h"
#include "datatypes.h"
#include "common/compute_tools.h"
#include "kernel/kernel.h"

points2image::points2image() :

	error_so_far(false),
	max_delta(0.0),
	pointcloud(),
	cameraExtrinsicMat(),
	cameraMat(),
	distCoeff(),
	imageSize(),
	results(),
	computeEnv(),
	computeProgram(),
	transformKernel(),
	maxCloudElementNo(0),
	pointcloudBuffer(),
	counterBuffer(),
	pixelBuffer()
#ifdef EPHOS_PINNED_MEMORY
	,pointcloudHostBuffer(),
	pointcloudStorage(nullptr),
	pixelHostBuffer(),
	pixelStorage(nullptr)
#endif
	{}

points2image::~points2image() {}

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
// 	try {
// 		datagen_file.open("../../../data/p2i_output.dat.gen", std::ios::binary);
// 	} catch (std::ofstream::failure) {
// 		std::cerr << "Error opening datagen file" << std::endl;
// 		exit(-2);
// 	}
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
	// create an opencl environment
	try {
	    std::vector<std::vector<std::string>> extensions = { {"cl_khr_fp64", "cl_amd_fp64" } };
	    computeEnv = ComputeTools::find_compute_platform(EPHOS_PLATFORM_HINT_S, EPHOS_DEVICE_HINT_S,
			EPHOS_DEVICE_TYPE_S, extensions);
	} catch (std::logic_error& e) {
	    std::cerr << "OpenCL setup failed. " << e.what() << std::endl;
	}
	//std::cout << "OpenCL platform: " << computeEnv.platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
	std::cout << "OpenCL device: " << computeEnv.device.getInfo<CL_DEVICE_NAME>() << std::endl;
	// compile opencl program and create the transformation kernel
	std::vector<cl::Kernel> kernels;
	try {
		std::vector<std::string> kernelNames({
			"pointcloud2image"
		});
		std::string sOptions =
#ifdef EPHOS_KERNEL_ATOMICS
			" -DEPHOS_KERNEL_ATOMICS"
#else
			""
#endif
#ifdef EPHOS_KERNEL_LOCAL_ATOMICS
			" -DEPHOS_KERNEL_LOCAL_ATOMICS"
#else
			""
#endif
#ifdef EPHOS_KERNEL_TRANSFORMS_PER_ITEM
			" -DEPHOS_KERNEL_TRANSFORMS_PER_ITEM=" STRINGIZE(EPHOS_KERNEL_TRANSFORMS_PER_ITEM)
#endif
			;
		std::string sSource(points2image_kernel_source_code);
		computeProgram = ComputeTools::build_program(computeEnv, sSource, sOptions,
			kernelNames, kernels);
	} catch (std::logic_error& e) {
		std::cerr << e.what() << std::endl;
		exit(EXIT_FAILURE);
	}
	transformKernel = kernels[0];

	// prepare the first iteration
	error_so_far = false;
	max_delta = 0.0;

	/*pointcloud = nullptr;
	cameraExtrinsicMat = nullptr;
	cameraMat = nullptr;
	distCoeff = nullptr;
	imageSize = nullptr;
	results = nullptr;*/
	maxCloudElementNo = 0;
	std::cout << "done" << std::endl;
}

void points2image::quit() {
	// close files
	try {
		input_file.close();
	} catch (std::ifstream::failure& e) {
	}
	try {
		output_file.close();

	} catch (std::ofstream::failure& e) {
	}
	try {
		datagen_file.close();
	} catch (std::ofstream::failure& e) {
	}


	if (maxCloudElementNo > 0) {
		// buffer cleanup
		//cl_int err = CL_SUCCESS;
		// free device buffers
		/*err = clReleaseMemObject(pointcloudBuffer);
		err = clReleaseMemObject(counterBuffer);
		err = clReleaseMemObject(pixelBuffer);*/
		pointcloudBuffer = cl::Buffer();
		counterBuffer = cl::Buffer();
		pixelBuffer = cl::Buffer();
#ifdef EPHOS_PINNED_MEMORY
		// free host buffers and memory
		/*err = clEnqueueUnmapMemObject(computeEnv.cmdqueue, pointcloudHostBuffer, pointcloudStorage,
			0, nullptr, nullptr);
		pointcloudStorage = nullptr;
		err = clReleaseMemObject(pointcloudHostBuffer);
		err = clEnqueueUnmapMemObject(computeEnv.cmdqueue, pixelHostBuffer, pixelStorage,
			0, nullptr, nullptr);
		pixelStorage = nullptr;
		err = clReleaseMemObject(pixelHostBuffer);*/
		computeEnv.cmdqueue.enqueueUnmapMemObject(pointcloudHostBuffer, pointcloudStorage);
		pointcloudHostBuffer = cl::Buffer();
		pointcloudStorage = nullptr;
		computeEnv.cmdqueue.enqueueUnmapMemObject(pixelHostBuffer, pixelStorage);
		pixelHostBuffer = cl::Buffer();
		pixelStorage = nullptr;
#endif // EPHOS_PINNED_MEMORY
	}
	// program cleanup
	/*clReleaseKernel(transformKernel);
	clReleaseProgram(computeProgram);
	clReleaseCommandQueue(computeEnv.cmdqueue);
	clReleaseContext(computeEnv.context);*/
	transformKernel = cl::Kernel();
	computeProgram = cl::Program();
	computeEnv.cmdqueue = cl::CommandQueue();
	computeEnv.device = cl::Device();
	computeEnv.context = cl::Context();
}

PointsImage points2image::cloud2Image(
	PointCloud& pointcloud,
	Mat44& cameraExtrinsicMat,
	Mat33& cameraMat,
	Vec5& distCoeff,
	ImageSize& imageSize) {

	// Prepare outputs data structures
	size_t imagePixelNo = imageSize.height*imageSize.width;
	PointsImage result = {
		new float[imagePixelNo], // intensity, will be free in read_next_testcases()
		new float[imagePixelNo], // distance
		new float[imagePixelNo], // min height
		new float[imagePixelNo], // max height
		-1, // max y
		imageSize.height, // min y
		imageSize.height, // result size height
		imageSize.width // result size width
	};
	std::memset(result.intensity, 0, sizeof(float)*imagePixelNo);
	std::memset(result.distance, 0, sizeof(float)*imagePixelNo);
	std::memset(result.min_height, 0, sizeof(float)*imagePixelNo);
	std::memset(result.max_height, 0, sizeof(float)*imagePixelNo);
	// prepare inputs buffers
	cl_int err = CL_SUCCESS;
	size_t pointNo = pointcloud.height*pointcloud.width;
	size_t cloudSize = pointNo*pointcloud.point_step*sizeof(float);
	// write cloud input to buffer
#ifdef EPHOS_ZERO_COPY
	//clEnqueueMapBuffer(computeEnv.cmdqueue, pixelHostBuffer,
//				CL_TRUE, CL_MAP_READ, 0, pixelSize, 0, nullptr, nullptr, &err);
	/*float* pointcloudStorage = (float*)clEnqueueMapBuffer(computeEnv.cmdqueue, pointcloudBuffer,
		CL_TRUE, CL_MAP_WRITE, 0, cloudSize, 0, nullptr, nullptr, &err);
	std::memcpy(pointcloudStorage, pointcloud.data, cloudSize);
	err = clEnqueueUnmapMemObject(computeEnv.cmdqueue, pointcloudBuffer, pointcloudStorage,
		0, nullptr, nullptr);*/
	float* pointcloudStorage = (float*)computeEnv.enqueueMapBuffer(pointcloudBuffer,
		CL_TRUE, CL_MAP_WRITE, 0, cloudSize);
	std::memcpy(pointcloudStorage, pointcloud.data, cloudSize);
	computeEnv.cmdqueue.enqueueUnmapMemObject(pointcloudBuffer, pointcloudStorage);
#else // !EPHOS_ZERO_COPY
	// pointcloud.data is pinned memory in the case of page-locked memory operation
	/*err = clEnqueueWriteBuffer(computeEnv.cmdqueue, pointcloudBuffer,
		CL_FALSE, 0, cloudSize, pointcloud.data, 0, nullptr, nullptr);*/
	computeEnv.cmdqueue.enqueueWriteBuffer(pointcloudBuffer, CL_FALSE,
		0, cloudSize, pointcloud.data);
#endif
	// transformation info for kernel
	double (*c)[4] = &cameraExtrinsicMat.data[0];
	TransformInfo transformInfo {
		{ 0.0, 0.0, 0.0, // initial rotation
			0.0, 0.0, 0.0,
			0.0, 0.0, 0.0 },
		{ 0.0, 0.0, 0.0 }, // initial translation
		{ cameraMat.data[0][0], cameraMat.data[1][1] }, // camera scale
		{ cameraMat.data[0][2] + 0.5, cameraMat.data[1][2] + 0.5}, // camera offset
		{ distCoeff.data[0], // distortion coefficients
			distCoeff.data[1],
			distCoeff.data[2],
			distCoeff.data[3],
			distCoeff.data[4] },
		{ imageSize.width, imageSize.height }, // image size
		pointcloud.width*pointcloud.height, // cloud point number
		(int)(pointcloud.point_step/sizeof(float)) // cloud point step
	};
	// calculate initial rotation and translation
	for (int row = 0; row < 3; row++) {
		for (int col = 0; col < 3; col++) {
			transformInfo.initRotation[row][col] = c[col][row];
			transformInfo.initTranslation[row] -= transformInfo.initRotation[row][col]*c[col][3];
		}
	}
	// set kernel parameters
	/*err = clSetKernelArg (transformKernel, 0, sizeof(TransformInfo),       &transformInfo);
	err = clSetKernelArg (transformKernel, 1, sizeof(cl_mem),    &pointcloudBuffer);
	err = clSetKernelArg(transformKernel, 2, sizeof(cl_mem), &pixelBuffer);
	#ifdef EPHOS_KERNEL_ATOMICS
	err = clSetKernelArg (transformKernel, 3, sizeof(cl_mem), &counterBuffer);
#ifdef EPHOS_KERNEL_LOCAL_ATOMICS
	err = clSetKernelArg(transformKernel, 4, sizeof(int), nullptr);
	err = clSetKernelArg(transformKernel, 5, sizeof(int), nullptr);
#endif // EPHOS_KERNEL_LOCAL_ATOMICS
#endif // EPHOS_KERNEL_ATOMICS*/
	transformKernel.setArg(0, transformInfo);
	transformKernel.setArg(1, pointcloudBuffer);
	transformKernel.setArg(2, pixelBuffer);
#ifdef EPHOS_KERNEL_ATOMICS
	transformKernel.setArg(3, counterBuffer);
#ifdef EPHOS_KERNEL_LOCAL_ATOMICS
	transformKernel.setArg(4, cl::Local(sizeof(int)));
	transformKernel.setArg(5, cl::Local(sizeof(int)));
#endif // EPHOS_KERNEL_LOCAL_ATOMICS
#endif // EPHOS_KERNEL_ATOMICS

	// initializing arriving point number
	int zero = 0;
	/*err = clEnqueueWriteBuffer(computeEnv.cmdqueue, counterBuffer, CL_FALSE,
		0, sizeof(int), &zero, 0, nullptr, nullptr);*/
	computeEnv.cmdqueue.enqueueWriteBuffer(counterBuffer, CL_FALSE,
		0, sizeof(int), &zero);
	// launch kernel on device
	size_t localRange = EPHOS_KERNEL_WORK_GROUP_SIZE;
#ifdef EPHOS_KERNEL_TRANSFORMS_PER_ITEM
	size_t globalRange = (pointNo/EPHOS_KERNEL_TRANSFORMS_PER_ITEM/localRange + 1)*localRange;
#else
	size_t globalRange = (pointNo/localRange + 1)*localRange;
#endif
	/*err = clEnqueueNDRangeKernel(computeEnv.cmdqueue, transformKernel, 1,
		nullptr,  &globalRange, &localRange, 0, nullptr, nullptr);*/
	cl::NDRange callOffset(0);
	cl::NDRange callLocal(localRange);
	cl::NDRange callGlobal(globalRange);
	computeEnv.cmdqueue.enqueueNDRangeKernel(transformKernel,
		callOffset, callGlobal, callLocal);

#ifdef EPHOS_KERNEL_ATOMICS
	int arrivingPixelNo;
	// read arriving pixel number from buffer
	//err = clEnqueueReadBuffer(computeEnv.cmdqueue, counterBuffer, CL_TRUE,
	//	0, sizeof(int), &arrivingPixelNo, 0, nullptr, nullptr);
	err = computeEnv.cmdqueue.enqueueReadBuffer(counterBuffer, CL_TRUE,
		0, sizeof(int), &arrivingPixelNo);
#else // !EPHOS_KERNEL_ATOMICS
	int arrivingPixelNo = pointNo;
#endif // !EPHOS_KERNEL_ATOMICS

	// read arriving pixels
/*#ifdef EPHOS_ZERO_COPY
	PixelData* pixelStorage = (PixelData*)clEnqueueMapBuffer(computeEnv.cmdqueue,
		pixelBuffer, CL_TRUE, CL_MAP_READ, 0, sizeof(PixelData)*arrivingPixelNo, 0, 0, nullptr, &err);
#elif EPHOS_PINNED_MEMORY
	err = clEnqueueReadBuffer(computeEnv.cmdqueue, pixelBuffer, CL_TRUE,
		0, sizeof(PixelData)*arrivingPixelNo, pixelStorage, 0, nullptr, nullptr);
#else // !EPHOS_ZERO_COPY && !EPHOS_PINNED_MEMORY
	PixelData* pixelStorage = new PixelData[arrivingPixelNo];
	err = clEnqueueReadBuffer(computeEnv.cmdqueue, pixelBuffer, CL_TRUE,
		0, sizeof(PixelData)*arrivingPixelNo, pixelStorage, 0, nullptr, nullptr);
#endif*/
#ifdef EPHOS_ZERO_COPY
	PixelData* pixelStorage = (PixelData*)computeEnv.cmdqueue.enqueueMapBuffer(pixelBuffer,
		CL_TRUE, CL_MAP_READ, 0, sizeof(PixelData)*arrivingPixelNo);
#elif EPHOS_PINNED_MEMORY
	computeEnv.cmdqueue.enqueueReadBuffer(pixelBuffer, CL_TRUE,
		0, sizeof(PixelData)*arrivingPixelNo, pixelStorage);
#else // !EPHOS_ZERO_COPY && !EPHOS_PINNED_MEMORY
	PixelData* pixelStorage = new PixelData[arrivingPixelNo];
	computeEnv.cmdqueue.enqueueReadBuffer(pixelBuffer, CL_TRUE,
		0, sizeof(PixelData)*arrivingPixelNo, pixelStorage);
#endif
	// process arriving pixels
	for (int j = 0; j < arrivingPixelNo; j++) {
		if (pixelStorage[j].position[0] > -1) {
			int iPixel = pixelStorage[j].position[1]*imageSize.width + pixelStorage[j].position[0];
			float currentDepth = result.distance[iPixel];
			float nextDepth = pixelStorage[j].depth*100.0f;

			if ((currentDepth == 0.0f) || (nextDepth <= currentDepth)) {
				float currentIntensity = result.intensity[iPixel];
				float nextIntensity = pixelStorage[j].intensity;
				// update intensity
				if ((currentDepth == nextDepth && nextIntensity > currentIntensity) ||
					(nextDepth < currentDepth) ||
					(currentDepth == 0)) {

					result.intensity[iPixel] = nextIntensity;
				}
				// update depth
				result.distance[iPixel] = nextDepth;
				// update height
				result.min_height[iPixel] = -1.25f;
				result.max_height[iPixel] = 0.0f;
				// update extends
				if (pixelStorage[j].position[1] > result.max_y) {
					result.max_y = pixelStorage[j].position[1];
				}
				if (pixelStorage[j].position[1] < result.min_y) {
					result.min_y = pixelStorage[j].position[1];
				}
			}
		}
	}
#ifdef EPHOS_ZERO_COPY
	computeEnv.cmdqueue.enqueueUnmapMemObject(pixelBuffer, pixelStorage);
#elif !defined(EPHOS_PINNED_MEMORY)
	delete[] pixelStorage;
#endif
	return result;
}



void points2image::prepare_compute_buffers(PointCloud& pointcloud) {
	int pointNo = pointcloud.height*pointcloud.width;
	size_t cloudSize = pointNo*pointcloud.point_step*sizeof(float);
	size_t pixelSize = pointNo*sizeof(PixelData);
	if (pointNo > maxCloudElementNo) {
		cl_int err = CL_SUCCESS;
		// free existing buffers
		if (maxCloudElementNo > 0) {
			// free device buffers
			/*err = clReleaseMemObject(pointcloudBuffer);
			err = clReleaseMemObject(counterBuffer);
			err = clReleaseMemObject(pixelBuffer);*/

#ifdef EPHOS_PINNED_MEMORY
			// free host buffers and memory
			/*err = clEnqueueUnmapMemObject(computeEnv.cmdqueue, pointcloudHostBuffer, pointcloudStorage,
				0, nullptr, nullptr);
			pointcloudStorage = nullptr;
			err = clReleaseMemObject(pointcloudHostBuffer);
			err = clEnqueueUnmapMemObject(computeEnv.cmdqueue, pixelHostBuffer, pixelStorage,
				0, nullptr, nullptr);
			pixelStorage = nullptr;
			err = clReleaseMemObject(pixelHostBuffer);*/
			computeEnv.cmdqueue.enqueueUnmapMemObject(pointcloudHostBuffer, pointcloudStorage);
			pointcloudStorage = nullptr;
			computeEnv.cmdqueue.enqueueUnmapMemObject(pixelHostBuffer, pixelStorage);
			pixelStorage = nullptr;
#endif // EPHOS_PINNED_MEMORY
		}
		{ // allocate new counter buffer on device
			cl_mem_flags flags = CL_MEM_READ_WRITE;
			//counterBuffer = clCreateBuffer(computeEnv.context, flags, sizeof(int), nullptr, &err);
			counterBuffer = cl::Buffer(computeEnv.context, flags, sizeof(int));
		}
		{ // allocate new buffers
			cl_mem_flags flags = CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY;
#ifdef EPHOS_ZERO_COPY
			flags |= CL_MEM_ALLOC_HOST_PTR;
#endif // EPHOS_ZERO_COPY
			//pointcloudBuffer = clCreateBuffer(computeEnv.context, flags, cloudSize, nullptr, &err);
			pointcloudBuffer = cl::Buffer(computeEnv.context, flags, cloudSize);
			flags = CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY;
#ifdef EPHOS_ZERO_COPY
			flags |= CL_MEM_ALLOC_HOST_PTR;
#endif // EPHOS_ZERO_COPY
			//pixelBuffer = clCreateBuffer(computeEnv.context, flags, pixelSize, nullptr, &err);
			pixelBuffer = cl::Buffer(computeEnv.context, flags, pixelSize);
		}
#ifdef EPHOS_PINNED_MEMORY
		{ // let opencl allocate host memory
			//cl_mem_flags flags = CL_MEM_HOST_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR;
			/*cl_mem_flags flags = CL_MEM_ALLOC_HOST_PTR | CL_MEM_HOST_WRITE_ONLY;
			pointcloudHostBuffer = clCreateBuffer(computeEnv.context, flags, cloudSize, nullptr, &err);
			flags = CL_MEM_ALLOC_HOST_PTR | CL_MEM_HOST_READ_ONLY;
			pixelHostBuffer = clCreateBuffer(computeEnv.context, flags, pixelSize, nullptr, &err);
			pointcloudStorage = (float*)clEnqueueMapBuffer(computeEnv.cmdqueue, pointcloudHostBuffer,
				CL_TRUE, CL_MAP_WRITE, 0, cloudSize, 0, nullptr, nullptr, &err);
			pixelStorage = (PixelData*)clEnqueueMapBuffer(computeEnv.cmdqueue, pixelHostBuffer,
				CL_TRUE, CL_MAP_READ, 0, pixelSize, 0, nullptr, nullptr, &err);*/
			cl_mem_flags flags = CL_MEM_ALLOC_HOST_PTR | CL_MEM_HOST_WRITE_ONLY;
			pointcloudHostBuffer = cl::Buffer(computeEnv.context, flags, cloudSize);
			flags = CL_MEM_ALLOC_HOST_PTR | CL_MEM_HOST_READ_ONLY;
			pixelHostBuffer = cl::Buffer(computeEnv.context, flags, pixelSize);
			pointcloudStorage = (float*)computeEnv.cmdqueue.enqueueMapBuffer(pointcloudHostBuffer,
				CL_TRUE, CL_MAP_WRITE, 0, cloudSize);
			pixelStorage = (PixelData*)computeEnv.cmdqueue.enqueueMapBuffer(pixelHostBuffer,
				CL_TRUE, CL_MAP_READ, 0, pixelSize);
		}
#endif // EPHOS_PINNED_MEMORY
		maxCloudElementNo = pointNo;
	}
#ifdef EPHOS_PINNED_MEMORY
	pointcloud.data = pointcloudStorage;
#else // !EPHOS_PINNED_MEMORY
	// manually allocate host memory
	// required in every step because it is freed again
	pointcloud.data = new float[pointNo*pointcloud.point_step];
#endif // !EPHOS_PINNED_MEMORY
}

