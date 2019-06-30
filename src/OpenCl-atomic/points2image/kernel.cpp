#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>	
#include <cstring>

#include "benchmark.h"
#include "datatypes.h"
#include "ocl/device/ocl_kernel.h"
#include "ocl/host/ocl_header.h"

#define STRINGIZE2(s) #s
#define STRINGIZE(s) STRINGIZE2(s)

// maximum allowed deviation from the reference results
#define MAX_EPS 0.001

// opencl platform hints
#if defined(EPHOS_PLATFORM_HINT)
#define EPHOS_PLATFORM_HINT_S STRINGIZE(EPHOS_PLATFORM_HINT)
#else 
#define EPHOS_PLATFORM_HINT_S ""
#endif

#if defined(EPHOS_DEVICE_HINT)
#define EPHOS_DEVICE_HINT_S STRINGIZE(EPHOS_DEVICE_HINT)
#else
#define EPHOS_DEVICE_HINT_S ""
#endif

#if defined(EPHOS_DEVICE_TYPE)
#define EPHOS_DEVICE_TYPE_S STRINGIZE(EPHOS_DEVICE_TYPE)
#else
#define EPHOS_DEVICE_TYPE_S ""
#endif

class points2image : public kernel {
private:
	// the number of testcases read
	int read_testcases = 0;
	// testcase and reference data streams
	std::ifstream input_file, output_file;
	// whether critical deviation from the reference data has been detected
	bool error_so_far = false;
	// deviation from the reference data
	double max_delta = 0.0;
	// the point clouds to process in one iteration
	PointCloud2* pointcloud2 = nullptr;
	// the associated camera extrinsic matrices
	Mat44* cameraExtrinsicMat = nullptr;
	// the associated camera intrinsic matrices
	Mat33* cameraMat = nullptr;
	// distance coefficients for the current iteration
	Vec5* distCoeff = nullptr;
	// image sizes for the current iteration
	ImageSize* imageSize = nullptr;
	// Algorithm results for the current iteration
	PointsImage* results = nullptr;
public:
	/*
	 * Initializes the kernel. Must be called before run().
	 */
	virtual void init();
	/**
	 * Performs the kernel operations on all input and output data.
	 * p: number of testcases to process in one step
	 */
	virtual void run(int p = 1);
	/**
	 * Finally checks whether all input data has been processed successfully.
	 */
	virtual bool check_output();
	
protected:
	/**
	* Reads the next test cases.
	* count: the number of testcases to read
	* returns: the number of testcases actually read
	*/
	virtual int read_next_testcases(int count);
	/**
	 * Compares the results from the algorithm with the reference data.
	 * count: the number of testcases processed 
	 */
	virtual void check_next_outputs(int count);
	/**
	 * Reads the number of testcases in the data set.
	 */
	int read_number_testcases(std::ifstream& input_file);
	
};
/**
 * Parses the next point cloud from the input stream.
 */
void  parsePointCloud(std::ifstream& input_file, PointCloud2* pointcloud2) {
	try {
		input_file.read((char*)&(pointcloud2->height), sizeof(int32_t));
		input_file.read((char*)&(pointcloud2->width), sizeof(int32_t));
		input_file.read((char*)&(pointcloud2->point_step), sizeof(uint32_t));
		pointcloud2->data = new float[pointcloud2->height * pointcloud2->width * pointcloud2->point_step];
		input_file.read((char*)pointcloud2->data, pointcloud2->height * pointcloud2->width * pointcloud2->point_step);
    }  catch (std::ifstream::failure) {
		throw std::ios_base::failure("Error reading the next point cloud.");
    }
}
/**
 * Parses the next camera extrinsic matrix.
 */
void  parseCameraExtrinsicMat(std::ifstream& input_file, Mat44* cameraExtrinsicMat) {
	try {
		for (int h = 0; h < 4; h++)
			for (int w = 0; w < 4; w++)
				input_file.read((char*)&(cameraExtrinsicMat->data[h][w]),sizeof(double));
	} catch (std::ifstream::failure) {
		throw std::ios_base::failure("Error reading the next extrinsic matrix.");		
	}
}
/**
 * Parses the next camera matrix.
 */
void parseCameraMat(std::ifstream& input_file, Mat33* cameraMat ) {
	try {
	for (int h = 0; h < 3; h++)
		for (int w = 0; w < 3; w++)
			input_file.read((char*)&(cameraMat->data[h][w]), sizeof(double));
	} catch (std::ifstream::failure) {
		throw std::ios_base::failure("Error reading the next camera matrix.");
    }
}
/**
 * Parses the next distance coefficients.
 */
void  parseDistCoeff(std::ifstream& input_file, Vec5* distCoeff) {
	try {
		for (int w = 0; w < 5; w++)
			input_file.read((char*)&(distCoeff->data[w]), sizeof(double));
	} catch (std::ifstream::failure) {
		throw std::ios_base::failure("Error reading the next set of distance coefficients.");
	}
}
/**
 * Parses the next image sizes.
 */
void  parseImageSize(std::ifstream& input_file, ImageSize* imageSize) {
	try {
		input_file.read((char*)&(imageSize->width), sizeof(int32_t));
		input_file.read((char*)&(imageSize->height), sizeof(int32_t));
	} catch (std::ifstream::failure) {
		throw std::ios_base::failure("Error reading the next image size.");
	}
}
/**
 * Parses the next reference image.
 */
void parsePointsImage(std::ifstream& output_file, PointsImage* goldenResult) {
	try {
		// read data of static size
		output_file.read((char*)&(goldenResult->image_width), sizeof(int32_t));
		output_file.read((char*)&(goldenResult->image_height), sizeof(int32_t));
		output_file.read((char*)&(goldenResult->max_y), sizeof(int32_t));
		output_file.read((char*)&(goldenResult->min_y), sizeof(int32_t));
		int pos = 0;
		int elements = goldenResult->image_height * goldenResult->image_width;
		goldenResult->intensity = new float[elements];
		goldenResult->distance = new float[elements];
		goldenResult->min_height = new float[elements];
		goldenResult->max_height = new float[elements];
		// read data of variable size
		for (int h = 0; h < goldenResult->image_height; h++)
			for (int w = 0; w < goldenResult->image_width; w++)
			{
				output_file.read((char*)&(goldenResult->intensity[pos]), sizeof(float));
				output_file.read((char*)&(goldenResult->distance[pos]), sizeof(float));
				output_file.read((char*)&(goldenResult->min_height[pos]), sizeof(float));
				output_file.read((char*)&(goldenResult->max_height[pos]), sizeof(float));
				pos++;
			}
	} catch (std::ios_base::failure) {
		throw std::ios_base::failure("Error reading the next reference image.");
	}
}

int points2image::read_next_testcases(int count)
{
	// free the memory that has been allocated in the previous iteration
	// and allocate new for the currently required data sizes
	if (pointcloud2) 
		for (int m = 0; m < count; ++m)
			delete [] pointcloud2[m].data;
	delete [] pointcloud2;
	pointcloud2 = new PointCloud2[count];
	delete [] cameraExtrinsicMat;
	cameraExtrinsicMat = new Mat44[count];
	delete [] cameraMat;
	cameraMat = new Mat33[count];
	delete [] distCoeff;
	distCoeff = new Vec5[count];
	delete [] imageSize;
	imageSize = new ImageSize[count];
	if (results)
	for (int m = 0; m < count; ++m)
	{
		delete [] results[m].intensity;
		delete [] results[m].distance;
		delete [] results[m].min_height;
		delete [] results[m].max_height;
	}
	delete [] results;
	results = new PointsImage[count];
	
	// iteratively read the data for the test cases
	int i;
	for (i = 0; (i < count) && (read_testcases < testcases); i++,read_testcases++)
	{
		try {
			parsePointCloud(input_file, pointcloud2 + i);
			parseCameraExtrinsicMat(input_file, cameraExtrinsicMat + i);
			parseCameraMat(input_file, cameraMat + i);
			parseDistCoeff(input_file, distCoeff + i);
			parseImageSize(input_file, imageSize + i);
		} catch (std::ios_base::failure& e) {
			std::cerr << e.what() << std::endl;
			exit(-3);
		}
	}

	return i;
}

int points2image::read_number_testcases(std::ifstream& input_file)
{
	// reads the number of testcases in the data stream
	int32_t number;
	try {
		input_file.read((char*)&(number), sizeof(int32_t));
	} catch (std::ifstream::failure) {
		throw std::ios_base::failure("Error reading the number of testcases.");
	}

	return number;
}
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
		std::string sSource(points2image_ocl_krnl);
		points2image_program = OCL_Tools::build_program(OCL_objs, sSource, std::string(""),
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

void points2image::check_next_outputs(int count)
{
	PointsImage reference;
	// parse the next reference image
	// and compare it to the data generated by the algorithm
	for (int i = 0; i < count; i++)
	{
		try {
			parsePointsImage(output_file, &reference);
		} catch (std::ios_base::failure& e) {
			std::cerr << e.what() << std::endl;
			exit(-3);
		}
		// detect image size deviation
		if ((results[i].image_height != reference.image_height)
			|| (results[i].image_width != reference.image_width))
		{
			error_so_far = true;
		}
		// detect image extend deviation
		if ((results[i].min_y != reference.min_y)
			|| (results[i].max_y != reference.max_y))
		{
			error_so_far = true;
		}
		// compare all pixels
		int pos = 0;
		for (int h = 0; h < reference.image_height; h++)
			for (int w = 0; w < reference.image_width; w++)
			{
				// compare members individually and detect deviations
				if (std::fabs(reference.intensity[pos] - results[i].intensity[pos]) > max_delta)
					max_delta = fabs(reference.intensity[pos] - results[i].intensity[pos]);
				if (std::fabs(reference.distance[pos] - results[i].distance[pos]) > max_delta)
					max_delta = fabs(reference.distance[pos] - results[i].distance[pos]);
				if (std::fabs(reference.min_height[pos] - results[i].min_height[pos]) > max_delta)
					max_delta = fabs(reference.min_height[pos] - results[i].min_height[pos]);
				if (std::fabs(reference.max_height[pos] - results[i].max_height[pos]) > max_delta)
					max_delta = fabs(reference.max_height[pos] - results[i].max_height[pos]);
				pos++;
			}
		// free the memory allocated by the reference image read above
		delete [] reference.intensity;
		delete [] reference.distance;
		delete [] reference.min_height;
		delete [] reference.max_height;
	}
}
bool points2image::check_output() {
	std::cout << "checking output \n";
	// complement to init()
	input_file.close();
	output_file.close();
	std::cout << "max delta: " << max_delta << "\n";
	if ((max_delta > MAX_EPS) || error_so_far) {
		return false;
	} else {
		return true;
	}
}
// set the external kernel instance used in main()
points2image a = points2image();
kernel& myKernel = a;
