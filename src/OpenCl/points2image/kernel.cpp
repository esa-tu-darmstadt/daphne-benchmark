#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>	
#include <cstring>

#include "benchmark.h"
#include "datatypes.h"
#include "stringify.h"
#include "ocl_header.h"

#define STRINGIZE2(s) #s
#define STRINGIZE(s) STRINGIZE2(s)

#define MAX_NUM_WORKITEMS 32
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

/**
 * Searches through the available OpenCL platforms to find one that suits the given arguments.
 * platformHint: platform name or index, empty for no restriction
 * deviceHint: device name or index, empty for no restriction
 * deviceType: can be one of ALL, CPU, GPU, ACC, DEF to only allow certaind devices
 * extensions: a chosen device must support at least one extension from each given extension set
 */
OCL_Struct find_compute_platform(
	std::string platformHint, std::string deviceHint, std::string deviceType,
	std::vector<std::vector<std::string>> extensions) {

	OCL_Struct result;
	cl_int errorCode = CL_SUCCESS;
	// query all platforms
	std::vector<cl_platform_id> availablePlatforms(16);
	cl_uint availablePlatformNo = 0;
	errorCode = clGetPlatformIDs(availablePlatforms.size(), availablePlatforms.data(), &availablePlatformNo);
	if (errorCode != CL_SUCCESS) {
		throw std::logic_error("Platform query failed: " + errorCode);
	}
	availablePlatforms.resize(availablePlatformNo);
	if (availablePlatforms.size() == 0) {
		throw std::logic_error("No platforms found");
	}
	// select a platform
	std::vector<cl_platform_id> selectedPlatforms;
	if (platformHint.length() > 0) {
		// select certain platforms
		int iPlatform;
		if (sscanf(platformHint.c_str(), "%d", &iPlatform) == 1) {
			// select platform by index
			if (iPlatform < availablePlatforms.size()) {
				selectedPlatforms.push_back(availablePlatforms[iPlatform]);
			} else {
				throw std::logic_error("Platform of index" + std::to_string(iPlatform) + " does not exist");
			}
			
		} else {
			// search for platforms that match a given name
			bool found = false;
			for (cl_platform_id p : availablePlatforms) {
				std::vector<char> nameBuffer(256);
				size_t nameLength = 0;
				errorCode = clGetPlatformInfo(p, CL_PLATFORM_NAME, nameBuffer.size(), nameBuffer.data(), &nameLength);
				if (errorCode != CL_SUCCESS) {
					// ignore for now
				} else {
					std::string platformName = std::string(nameBuffer.data(), nameLength);
					if (platformName.find(platformHint) != std::string::npos) {
						selectedPlatforms.push_back(p);
						found = true;
					}
				}
			}
			if (!found) {
				throw std::logic_error("No platform that matches " + platformHint);
				// TODO: display any query errors
			}
		}
	} else {
		// consider all platforms
		for (cl_platform_id p : availablePlatforms) {
			selectedPlatforms.push_back(p);
		}
	}
	// query devices
	// filter devices by type
	std::vector<cl_device_id> filteredDevices;
	// detect the device type
	cl_device_type type = CL_DEVICE_TYPE_ALL;
	if (deviceType.find("CPU") != std::string::npos) {
		type = CL_DEVICE_TYPE_CPU;
	} else if (deviceType.find("GPU") != std::string::npos) {
		type = CL_DEVICE_TYPE_GPU;
	} else if (deviceType.find("ACC") != std::string::npos) {
		type = CL_DEVICE_TYPE_ACCELERATOR;
	} else if (deviceType.find("DEF") != std::string::npos) {
		type = CL_DEVICE_TYPE_DEFAULT;
	}
	std::ostringstream sQueryError;
	bool errorDetected = false;
	// filter devices
	for (cl_platform_id p : selectedPlatforms) {
	std::vector<cl_device_id> devices(16);
	cl_uint deviceNo = 0;
	errorCode = clGetDeviceIDs(p, type, devices.size(), devices.data(), &deviceNo);
	if (errorCode != CL_SUCCESS) {
		sQueryError << "Failed to get devices (" << errorCode << ")" << std::endl;
		errorDetected = true;
	}
	devices.resize(deviceNo);
		for (cl_device_id d : devices) {
			filteredDevices.push_back(d);
		}
	}
	if (filteredDevices.size() == 0) {
		std::ostringstream sError;
		sError << "No devices found.";
		if (errorDetected) {
			sError << " Failed queries:" << std::endl;
			sError << sQueryError.str();
		}
		throw std::logic_error(sError.str());
	}
	// select devices
	std::vector<cl_device_id> selectedDevices;
	if (deviceHint.length() > 0) {
		// select specific devices
	int iDevice;
	if (sscanf(deviceHint.c_str(), "%d", &iDevice) == 1) {
		// select by index
		if (iDevice < filteredDevices.size()) {
			selectedDevices.push_back(filteredDevices[iDevice]);
		} else {
			throw std::logic_error("Device of index " + std::to_string(iDevice) + " does not exist");
		}
	} else {
		// select by name
		bool found = false;
		bool queryError = false;
		std::ostringstream sQueryError;
		for (cl_device_id d : filteredDevices) {
			std::vector<char> nameBuffer(256);
			size_t nameLength = 0;
			errorCode = clGetDeviceInfo(d, CL_DEVICE_NAME, nameBuffer.size(), nameBuffer.data(), &nameLength);
			if (errorCode != CL_SUCCESS) {
				sQueryError << "Failed to get device name (" << errorCode << ")" << std::endl;
				queryError = true;
			} else {
				std::string deviceName(nameBuffer.data(), nameLength);
				//std::string deviceName = d.getInfo<CL_DEVICE_NAME>();
				if (deviceName.find(deviceHint) != std::string::npos) {
					selectedDevices.push_back(d);
					found = true;
				}
			}
		} 
		if (!found) {
			std::ostringstream sError;
			sError << "No device that matches " << deviceHint << std::endl;
			if (queryError) {
				sError << "Failed device queries:" << std::endl;
				sError << sQueryError.str();
			}
			throw std::logic_error(sError.str());
		}
	}
	} else {
		// select all devices
		for (cl_device_id d : filteredDevices) {
			selectedDevices.push_back(d);
		}
	}
	// filter by extensions
	std::vector<cl_device_id> supportedDevices;
	if (extensions.size() > 0) {
		// request at least one extension
		bool found = false;
		bool queryError = false;
		std::ostringstream sQueryError;
		for (cl_device_id d : selectedDevices) {
			std::vector<char> extensionBuffer(4096);
			size_t extensionLength = 0;
			errorCode = clGetDeviceInfo(d, CL_DEVICE_EXTENSIONS, extensionBuffer.size(), extensionBuffer.data(), &extensionLength);
			if (errorCode != CL_SUCCESS) {
				sQueryError << "Failed to get device extensions (" << errorCode << ")" << std::endl;
				queryError = true;
			} else {
				std::string supportedExtensions(extensionBuffer.data(), extensionLength);
				// for each extension set at least one extension must be supported
				bool deviceSupported = true;
				for (std::vector<std::string> extensionSet : extensions) {
					bool extFound = false;
					for (std::string ext : extensionSet) {
						if (supportedExtensions.find(ext) != std::string::npos) {
							extFound = true;
						}
					}
					if (!extFound) {
						deviceSupported = false;
					}
				}
				if (deviceSupported) {
					supportedDevices.push_back(d);
				}
			}
		}
		if (supportedDevices.size() == 0) {
			std::ostringstream sError;
			sError << "No device found that supports the required extensions: " << std::endl;
			for (std::vector<std::string> extensionSet : extensions) {
				sError << "{ ";
				for (std::string ext : extensionSet) {
					sError << ext << " ";
				}
				sError << "} ";
			}
			sError << std::endl;
			if (queryError) {
				sError << sQueryError.str();
			}
			throw std::logic_error(sError.str());
		}
	} else {
		// all devices pass
		for (cl_device_id d : selectedDevices) {
			supportedDevices.push_back(d);
		}
	}
	// create context and queue
	// select the first supported device
	result.device = supportedDevices[0];
	result.context = clCreateContext(nullptr, 1, &supportedDevices[0], nullptr, nullptr, &errorCode);
	if (errorCode != CL_SUCCESS) {
		throw std::logic_error("Context creation failed: " + std::to_string(errorCode));
	}
	cl_command_queue_properties queueProperties = 0;
	result.cmdqueue = clCreateCommandQueue(result.context, supportedDevices[0], queueProperties, &errorCode);
	if (errorCode != CL_SUCCESS) {
		throw std::logic_error("Command queue creation failed: " + std::to_string(errorCode));
	}
	return result;
}

	

void points2image::run(int p) {
	// do not measure setup time
	pause_func();
	OCL_Struct OCL_objs;
	try {
	    std::vector<std::vector<std::string>> extensions = { {"cl_khr_fp64", "cl_amd_fp64" } };
	    OCL_objs = find_compute_platform(EPHOS_PLATFORM_HINT_S, EPHOS_DEVICE_HINT_S, 
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
	// constructing the OpenCL program for the points2image function
	cl_int err;
	cl_program points2image_program = clCreateProgramWithSource(OCL_objs.context, 1, (const char **)&points2image_ocl_krnl, NULL, &err);
	// building the OpenCL program for all the objects
	err = clBuildProgram(points2image_program, 1, &OCL_objs.device, NULL, NULL, NULL);
	if (err != CL_SUCCESS) {
		std::vector<char> logBuffer(8192);
		size_t logLength = 0;
		err = clGetProgramBuildInfo(points2image_program, OCL_objs.device, 
			CL_PROGRAM_BUILD_LOG, logBuffer.size(), logBuffer.data(), &logLength);
		std::string log(logBuffer.data(), logLength);
		std::cerr << "Build failed" << std::endl;
		std::cerr << log << std::endl;
		exit(EXIT_FAILURE);
	}
	// kernel
	cl_kernel points2image_kernel = clCreateKernel(points2image_program, "pointcloud2_to_image", &err);
	// getting max workgroup size
	size_t local_size;
	err = clGetDeviceInfo(OCL_objs.device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &local_size, NULL);
	// min total number of threads for executing the kernel
	cl_uint compute_unit;
	err =  clGetDeviceInfo(OCL_objs.device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &compute_unit, NULL);
	size_t global_size = compute_unit * local_size;
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
			size_t pc2data_numelements = pointcloud2[i].height * pointcloud2[i].width * pointcloud2[i].point_step;
			size_t size_pc2data = pc2data_numelements * sizeof(float);
			// Creating zero-copy buffer for pointcloud data using "CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR"
			cl_mem buff_pointcloud2_data =  clCreateBuffer(OCL_objs.context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, size_pc2data, NULL, &err);
			// Enqueuing mapbuffer to put the input data buff_pointcloud2_data on the map region between host and device
			float* tmp_pointcloud2_data = (float*) clEnqueueMapBuffer(OCL_objs.cmdqueue,
				buff_pointcloud2_data, CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, 0,
				size_pc2data, 0, 0, NULL, &err);

			// Copying from host memory to pinned host memory which is used by the CVengine automatically
			for (uint j=0; j<pc2data_numelements; j++) {
				tmp_pointcloud2_data[j] = pointcloud2[i].data[j];
			}
			// Unmapping the pointer, this will return the control to the device
			clEnqueueUnmapMemObject(OCL_objs.cmdqueue, buff_pointcloud2_data, tmp_pointcloud2_data, 0, NULL, NULL);
			// Prepare outputs buffers
			size_t outbuff_numelements = imageSize[i].height*imageSize[i].width;
			size_t size_outputbuff = outbuff_numelements * sizeof(float);
			// Allocate space in host to store results comming from GPU
			// These will be freed in read_next_testcases()
			results[i].intensity  = new float[outbuff_numelements];
			std::memset(results[i].intensity, 0, sizeof(float)*outbuff_numelements);
			results[i].distance   = new float[outbuff_numelements];
			std::memset(results[i].distance, 0, sizeof(float)*outbuff_numelements);
			results[i].min_height = new float[outbuff_numelements];
			std::memset(results[i].min_height, 0, sizeof(float)*outbuff_numelements);
			results[i].max_height = new float[outbuff_numelements];
			std::memset(results[i].max_height, 0, sizeof(float)*outbuff_numelements);
			// Creating zero-copy buffers for pids data
			cl_mem buff_pids        = clCreateBuffer(OCL_objs.context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, pointcloud2[i].width * sizeof(int),   NULL, &err);
			cl_mem buff_enable_pids = clCreateBuffer(OCL_objs.context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, pointcloud2[i].width * sizeof(int),   NULL, &err);
			cl_mem buff_pointdata2  = clCreateBuffer(OCL_objs.context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, pointcloud2[i].width * sizeof(float), NULL, &err);
			cl_mem buff_intensity   = clCreateBuffer(OCL_objs.context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, pointcloud2[i].width * sizeof(float), NULL, &err);
			cl_mem buff_py          = clCreateBuffer(OCL_objs.context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, pointcloud2[i].width * sizeof(int),   NULL, &err);
			// Set kernel parameters
			err = clSetKernelArg (points2image_kernel, 0, sizeof(int),       &pointcloud2[i].height);
			err = clSetKernelArg (points2image_kernel, 1, sizeof(int),       &pointcloud2[i].width);
 			err = clSetKernelArg (points2image_kernel, 2, sizeof(int),       &pointcloud2[i].point_step);
			err = clSetKernelArg (points2image_kernel, 3, sizeof(cl_mem),    &buff_pointcloud2_data);
			
			Mat44 tmp_cameraExtrinsic;
			Mat33 tmp_cameraMat;
			Vec5  tmp_distCoeff;

			for (uint p=0; p<4; p++){
				for (uint q=0; q<4; q++) {
					tmp_cameraExtrinsic.data[p][q] = cameraExtrinsicMat[i].data[p][q];
				}
			}
			for (uint p=0; p<3; p++){
				for (uint q=0; q<3; q++) {
						tmp_cameraMat.data[p][q] = cameraMat[i].data[p][q];
				}
			}
			for (uint p=0; p<5; p++){
					tmp_distCoeff.data[p] = distCoeff[i].data[p];
			}

			err = clSetKernelArg (points2image_kernel, 4,  sizeof(Mat44),  &tmp_cameraExtrinsic);
			err = clSetKernelArg (points2image_kernel, 5,  sizeof(Mat33),  &tmp_cameraMat);
			err = clSetKernelArg (points2image_kernel, 6,  sizeof(Vec5),   &tmp_distCoeff);

			err = clSetKernelArg (points2image_kernel, 7,  sizeof(ImageSize), &imageSize[i]);
			err = clSetKernelArg (points2image_kernel, 8,  sizeof(cl_mem), &buff_pids);
			err = clSetKernelArg (points2image_kernel, 9,  sizeof(cl_mem), &buff_enable_pids);
			err = clSetKernelArg (points2image_kernel, 10, sizeof(cl_mem), &buff_pointdata2);
			err = clSetKernelArg (points2image_kernel, 11, sizeof(cl_mem), &buff_intensity);
			err = clSetKernelArg (points2image_kernel, 12, sizeof(cl_mem), &buff_py);

			// Update global size
			size_t tmp_size =  pointcloud2[i].width / local_size;
			global_size = (tmp_size + 1) * /*MAX_NUM_WORKITEMS*/ local_size; // ~ 50000
			// Launch kernel on device
			err = clEnqueueNDRangeKernel(OCL_objs.cmdqueue, points2image_kernel, 1, NULL,  &global_size, &local_size, 0, NULL, NULL);
			// CPU update of msg_intensity, msg_distance, msg_min_height, msg_max_height, etc
			size_t nelems_tmp     = pointcloud2[i].width;
			size_t size_tmp_int   = nelems_tmp * sizeof(int);
			size_t size_tmp_float = nelems_tmp * sizeof(float);

			int* cpu_pids = (int*) clEnqueueMapBuffer(OCL_objs.cmdqueue, buff_pids, 
				CL_TRUE, CL_MAP_READ, 0, size_tmp_int, 0, 0, NULL, &err);
			int* cpu_enable_pids = (int*) clEnqueueMapBuffer(OCL_objs.cmdqueue, buff_enable_pids, 
				CL_TRUE, CL_MAP_READ, 0, size_tmp_int, 0, 0, NULL, &err);
			float* cpu_pointdata2 = (float*) clEnqueueMapBuffer(OCL_objs.cmdqueue, buff_pointdata2,
				CL_TRUE, CL_MAP_READ, 0,size_tmp_float, 0, 0, NULL, &err);
			float* cpu_intensity = (float*) clEnqueueMapBuffer(OCL_objs.cmdqueue, buff_intensity,
				CL_TRUE, CL_MAP_READ, 0, size_tmp_float, 0, 0, NULL, &err);
			int* cpu_py = (int*) clEnqueueMapBuffer(OCL_objs.cmdqueue, buff_py, 
				CL_TRUE, CL_MAP_READ, 0, size_tmp_int, 0, 0, NULL, &err);
 
			// transfer image size
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
			for (unsigned int y = 0; y < pc2_height; ++y) {
				for (unsigned int x = 0; x < pc2_width; x++) {
					if (cpu_enable_pids[x] == 1) {
						int pid = cpu_pids [x];
						float tmp_pointdata2 = cpu_pointdata2[x] * 100;
						float tmp_distance = results[i].distance[pid];

						bool cond1 = (tmp_distance == 0.0f);
						bool cond2 = (tmp_distance >= tmp_pointdata2);
						if( cond1 || cond2 ) {
							bool cond3 = (tmp_distance == tmp_pointdata2);
							bool cond4 = (results[i].intensity[pid] <  cpu_intensity[x]);
							bool cond5 = (tmp_distance >  tmp_pointdata2);
							bool cond6 = (tmp_distance == 0);

							if ((cond3 && cond4) || cond5 || cond6) {
								results[i].intensity[pid] = cpu_intensity[x];
							}
							results[i].distance[pid]  = float(tmp_pointdata2);
							int tmp_py = cpu_py[x];
							results[i].max_y = tmp_py > results[i].max_y ? tmp_py : results[i].max_y;
							results[i].min_y = tmp_py < results[i].min_y ? tmp_py : results[i].min_y;
						}
						results[i].min_height[pid] = -1.25f;
						results[i].max_height[pid] = 0.0f;
						
					} 
				}
			}
			// cleanup
			clEnqueueUnmapMemObject(OCL_objs.cmdqueue, buff_pids, cpu_pids, 0, NULL, NULL);
			clEnqueueUnmapMemObject(OCL_objs.cmdqueue, buff_enable_pids, cpu_enable_pids, 0, NULL, NULL);
			clEnqueueUnmapMemObject(OCL_objs.cmdqueue, buff_pointdata2, cpu_pointdata2, 0, NULL, NULL);
			clEnqueueUnmapMemObject(OCL_objs.cmdqueue, buff_intensity, cpu_intensity, 0, NULL, NULL);
			clEnqueueUnmapMemObject(OCL_objs.cmdqueue, buff_py, cpu_py, 0, NULL, NULL);

			clReleaseMemObject(buff_pointcloud2_data);
			clReleaseMemObject(buff_pids);
			clReleaseMemObject(buff_enable_pids);
			clReleaseMemObject(buff_pointdata2);
			clReleaseMemObject(buff_intensity);
			clReleaseMemObject(buff_py);
		}
		pause_func();
		check_next_outputs(count);
	}
	// cleanup
	err = clReleaseKernel(points2image_kernel);
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
