#include "benchmark.h"
#include "datatypes.h"
#include <math.h>
#include <iostream>
#include <fstream>

#define MAX_EPS 0.001

class points2image : public kernel {
    public:
        virtual void init();
        virtual void run(int p = 1);
        virtual bool check_output();
        PointCloud2* pointcloud2        = NULL;
        Mat44*       cameraExtrinsicMat = NULL;
        Mat33*       cameraMat          = NULL;
        Vec5*        distCoeff          = NULL;
        ImageSize*   imageSize          = NULL;
        PointsImage* results            = NULL;
    protected:
        virtual int  read_next_testcases(int count);
        virtual void check_next_outputs(int count);
        int read_number_testcases(std::ifstream& input_file);
        int read_testcases = 0;
        std::ifstream input_file, output_file;
        bool error_so_far;
        double max_delta;
};

int points2image::read_number_testcases(std::ifstream& input_file)
{
    int32_t number;
    try {
	input_file.read((char*)&(number), sizeof(int32_t));
    } catch (std::ifstream::failure e) {
	std::cerr << "Error reading file\n";
	exit(-3);
    }

    return number;    
}


void  parsePointCloud(std::ifstream& input_file, PointCloud2* pointcloud2) {
    try {
    	input_file.read((char*)&(pointcloud2->height), sizeof(int32_t));
    	input_file.read((char*)&(pointcloud2->width), sizeof(int32_t));
    	input_file.read((char*)&(pointcloud2->point_step), sizeof(uint32_t));

    	// Not used
    	/*int pos = 0;*/

    	pointcloud2->data = new float[pointcloud2->height * pointcloud2->width * pointcloud2->point_step];
    	input_file.read((char*)pointcloud2->data, pointcloud2->height * pointcloud2->width * pointcloud2->point_step);
    }
    catch (std::ifstream::failure e) {
	std::cerr << "Error reading file\n";
	exit(-3);
    }
}

void  parseCameraExtrinsicMat(std::ifstream& input_file, Mat44* cameraExtrinsicMat) {
    try{
        for (int h = 0; h < 4; h++)
            for (int w = 0; w < 4; w++)
    	        input_file.read((char*)&(cameraExtrinsicMat->data[h][w]),sizeof(double));
    }
    catch (std::ifstream::failure e) {
        std::cerr << "Error reading file\n";
	exit(-3);
    }
}

void parseCameraMat(std::ifstream& input_file, Mat33* cameraMat ) {
    try {
        for (int h = 0; h < 3; h++)
            for (int w = 0; w < 3; w++)
    	        input_file.read((char*)&(cameraMat->data[h][w]), sizeof(double));
    }
    catch (std::ifstream::failure e) {
        std::cerr << "Error reading file\n";
        exit(-3);
    }
}

void  parseDistCoeff(std::ifstream& input_file, Vec5* distCoeff) {
    try {
        for (int w = 0; w < 5; w++)
	    input_file.read((char*)&(distCoeff->data[w]), sizeof(double));
    }
    catch (std::ifstream::failure e) {
        std::cerr << "Error reading file\n";
	exit(-3);
    }
}

void  parseImageSize(std::ifstream& input_file, ImageSize* imageSize) {
    try {
        input_file.read((char*)&(imageSize->width), sizeof(int32_t));
        input_file.read((char*)&(imageSize->height), sizeof(int32_t));
    }
    catch (std::ifstream::failure e) {
        std::cerr << "Error reading file\n";
	exit(-3);
    }
  }

void parsePointsImage(std::ifstream& output_file, PointsImage* goldenResult) {
    try {
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
        for (int h = 0; h < goldenResult->image_height; h++)
            for (int w = 0; w < goldenResult->image_width; w++)
            {
	        output_file.read((char*)&(goldenResult->intensity[pos]), sizeof(float));
	        output_file.read((char*)&(goldenResult->distance[pos]), sizeof(float));
	        output_file.read((char*)&(goldenResult->min_height[pos]), sizeof(float));
	        output_file.read((char*)&(goldenResult->max_height[pos]), sizeof(float));
	        pos++;
            }
    }
    catch (std::ifstream::failure e) {
        std::cerr << "Error reading file\n";
        exit(-3);
    }
}

// return how many could be read
int points2image::read_next_testcases(int count)
{
    int i;

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

    for (i = 0; (i < count) && (read_testcases < testcases); i++,read_testcases++)
    {
        parsePointCloud(input_file, pointcloud2 + i);
        parseCameraExtrinsicMat(input_file, cameraExtrinsicMat + i);
        parseCameraMat(input_file, cameraMat + i);
        parseDistCoeff(input_file, distCoeff + i);
        parseImageSize(input_file, imageSize + i);
    }

    return i;
}

void points2image::init() {

    std::cout << "init\n";
    //input_file.read((char*)&testcases, sizeof(uint32_t));
    testcases = /*2500*/ 2500;

    #if defined (PRINTINFO)
    std::cout << "# testcases reduced to " << testcases << " (only for fast debugging)." << std::endl;
    #endif

    input_file.exceptions  ( std::ifstream::failbit | std::ifstream::badbit );
    output_file.exceptions ( std::ifstream::failbit | std::ifstream::badbit );
    try {
        input_file.open("../../../data/p2i_input.dat", std::ios::binary);

	// For Renesas OpenCL: use singleprec ONLY
	//output_file.open("data/output.dat", std::ios::binary);
	#if defined (OPENCL)
	output_file.open("../../../data/p2i_output.dat", std::ios::binary);
	#else
	output_file.open("../../../data/output_doubleprec.dat", std::ios::binary);
	#endif
    }
    catch (std::ifstream::failure e) {
        std::cerr << "Error opening file\n";
        exit(-2);
    }

    error_so_far = false;
    max_delta = 0.0;

    testcases = read_number_testcases(input_file);
    
    pointcloud2 = NULL;
    cameraExtrinsicMat = NULL;
    cameraMat = NULL;
    distCoeff = NULL;
    imageSize = NULL;
    results = NULL;

    std::cout << "done\n" << std::endl;
}


/**
    Helperfunction which allocates and sets everything to a given value.
*/
float* assign(uint32_t count, float value) {
    float* result;

    result = new float[count];
    for (int i = 0; i < count; i++) {
        result[i] = value;
    }
    return result;
}

/**
   This code is extracted from Autoware, file:
   ~/Autoware/ros/src/sensing/fusion/packages/points2image/lib/points_image/points_image.cpp
*/

#if defined (OPENCL)

#else
	// Print serial
	//#define PRINT_CPP
	//#define PRINT_CRITICAL
	//#define IT_CRITICAL 45285
#endif

PointsImage
pointcloud2_to_image(const PointCloud2& pointcloud2,
                     const Mat44& cameraExtrinsicMat,
                     const Mat33& cameraMat, const Vec5& distCoeff,
                     const ImageSize& imageSize)
{
        int w = imageSize.width;
        int h = imageSize.height;

	#if defined (PRINT_CPP)
	std::cout << "w : " << w << " | h : " << h << " | pc2_width : " << pointcloud2.width << " | pc2_height : " << pointcloud2.height << " | pc2_pstep : " << pointcloud2.point_step << std::endl;
	#endif

        uintptr_t cp = (uintptr_t)pointcloud2.data;

        PointsImage msg;

	msg.intensity = assign(w * h, 0);
	msg.distance = assign(w * h, 0);
        msg.min_height = assign(w * h, 0);
        msg.max_height = assign(w * h, 0);

        Mat33 invR;
	Mat13 invT;
	// invR= cameraExtrinsicMat(cv::Rect(0,0,3,3)).t();
	for (int row = 0; row < 3; row++)
	  for (int col = 0; col < 3; col++)
	    invR.data[row][col] = cameraExtrinsicMat.data[col][row];
	for (int row = 0; row < 3; row++) {
	  invT.data[row] = 0.0;
	  for (int col = 0; col < 3; col++)
	    //invT = -invR*(cameraExtrinsicMat(cv::Rect(3,0,1,3)));
	    invT.data[row] -= invR.data[row][col] * cameraExtrinsicMat.data[col][3];
	}

	#if defined (PRINT_CPP)
	std::cout << "(Mat33) invR: " << std::endl;
	for (unsigned int idx = 0; idx < 3; idx ++) {
		for (unsigned int idy = 0; idy < 3; idy ++) {
			std::cout << idx << " | "<< idy << " | "<< invR.data[idx][idy] << std::endl;
		}
	}
	std::cout << std::endl;

	std::cout << "(Mat13) invT: " << std::endl;
	for (unsigned int idx = 0; idx < 3; idx ++) {
		std::cout << idx << " | " << invT.data[idx] << std::endl;
	}
	std::cout << std::endl;
	#endif

        msg.max_y = -1;
        msg.min_y = h;
        msg.image_height = imageSize.height;
        msg.image_width = imageSize.width;

	#if defined (PRINT_CPP)
	std::cout << "msg.max_y : " << msg.max_y << " | msg.min_y : " << msg.min_y << " | msg.image_height : " << msg.image_height << " | msg.image_width : " << msg.image_width << std::endl;
	#endif

	#if defined (PRINT_CPP)
	printf("\ncp: 0 - 31\n");
	for (unsigned int idx = 0; idx < 32; idx ++) {
		printf("%-5u %10f\n", idx, pointcloud2.data[idx]);}
	printf("\ncp: 32 - 63\n");
	for (unsigned int idx = 32; idx < 64; idx ++) {
		printf("%-5u %10f\n", idx, pointcloud2.data[idx]);}
	#endif



       for (uint32_t y = 0; y < pointcloud2.height; ++y) {
                for (uint32_t x = 0; x < pointcloud2.width; ++x) {
                        float* fp = (float *)(cp + (x + y*pointcloud2.width) * pointcloud2.point_step);
                        double intensity = fp[4];

                        Mat13 point, point2;
                        point2.data[0] = double(fp[0]);
                        point2.data[1] = double(fp[1]);
                        point2.data[2] = double(fp[2]);

                        //point = point * invR.t() + invT.t();
			for (int row = 0; row < 3; row++) {
			  point.data[row] = invT.data[row];
			  for (int col = 0; col < 3; col++) 
			    point.data[row] += point2.data[col] * invR.data[row][col];
			}


			#if defined (PRINT_CPP)
			printf("\n\n");
			printf("%-15s %10u\n", "inner loop id (x) : ", x);
			printf("%-15s %15u\n", "offset1 : ", (x + y*pointcloud2.width) * pointcloud2.point_step);
			//printf("%-35s %10u\n", "pointcloud2.data : ", pointcloud2.data);
			//printf("%-35s %10u\n", "cp : ", cp);

			//printf("%-35s %10u\n", "pointcloud2.data + offset1 : ", pointcloud2.data +  (x + y*pointcloud2.width) * pointcloud2.point_step);
			//printf("%-35s %10u\n", "cp + offset1 : ", cp+  (x + y*pointcloud2.width) * pointcloud2.point_step);

			printf("\n");
			for (unsigned int k = 0; k < 5; k ++) {
				printf("%s %u %s %15f\n", "fp[", k, "] : ", fp[k]);
				//printf("%s %u %s %15f\n", "fp*[", k, "] : ", pointcloud2.data[(x + y*pointcloud2.width) * pointcloud2.point_step + k]);
			}

			printf("%s %15f\n", "intensity: ", intensity);

			printf("\n");
			for (unsigned int k = 0; k < 3; k ++) {
				printf("%s %u %s %15f\n", "point2.data[", k, "] : ", point2.data[k]);
			}

			printf("\n");
			for (unsigned int k = 0; k < 3; k ++) {
				printf("%s %u %s %15f\n", "point.data[", k, "] : ", point.data[k]);
			}
			#endif

                        if (point.data[2] <= 2.5) {
				#if defined (PRINT_CPP)
				printf("\nSkipping rest of iteration x # %u\n", x);
				#endif
                                continue;
                        }

			#if defined (PRINT_CPP)
			printf("\nProcessing iteration x # %u\n", x);
			#endif

                        double tmpx = point.data[0] / point.data[2];
                        double tmpy = point.data[1]/ point.data[2];
                        double r2 = tmpx * tmpx + tmpy * tmpy;
                        double tmpdist = 1 + distCoeff.data[0] * r2
                                + distCoeff.data[1] * r2 * r2
                                + distCoeff.data[4] * r2 * r2 * r2;

			#if defined (PRINT_CPP)
			printf("\n");
			printf("%-25s %10f\n", "tmpx : ", tmpx);
			printf("%-25s %10f\n", "tmpy : ", tmpy);
			printf("%-25s %10f\n", "r2 : ", r2);
			printf("%-25s %10f\n", "tmpdist : ", tmpdist);
			#endif

                        Point2d imagepoint;
                        imagepoint.x = tmpx * tmpdist
                                + 2 * distCoeff.data[2] * tmpx * tmpy
                                + distCoeff.data[3] * (r2 + 2 * tmpx * tmpx);
                        imagepoint.y = tmpy * tmpdist
                                + distCoeff.data[2] * (r2 + 2 * tmpy * tmpy)
                                + 2 * distCoeff.data[3] * tmpx * tmpy;
                        imagepoint.x = cameraMat.data[0][0] * imagepoint.x + cameraMat.data[0][2];
                        imagepoint.y = cameraMat.data[1][1] * imagepoint.y + cameraMat.data[1][2];

                        int px = int(imagepoint.x + 0.5);
                        int py = int(imagepoint.y + 0.5);

			#if defined (PRINT_CPP)
			printf("\n");
			printf("%-25s %10f\n", "imagepoint.x : ", imagepoint.x);
			printf("%-25s %10f\n", "imagepoint.y : ", imagepoint.y);
			printf("%-25s %10i\n", "px : ", px);
			printf("%-25s %10i\n", "py : ", py);
			#endif

                        if(0 <= px && px < w && 0 <= py && py < h)
                        {
                                int pid = py * w + px; 

				/*
				#if defined (PRINT_CRITICAL)
				if (x == IT_CRITICAL) {
					printf("x=%u\n", x);
					printf("%-25s %10i\n", "tmp_gmem : ", msg.distance[pid]);
				}
				#endif
				*/

				#if defined (PRINT_CRITICAL)
				printf("px & py within [0, w&h> range, happens when x=%u, px=%i, py=%i, pid=%i, msg.distance[pid]=%f\n", x, px, py, pid, msg.distance[pid]);
				#endif


				// Confirmed: pid can be zero
				// if (pid == 0) {printf("pid==0\n");};

                                if(msg.distance[pid] == 0 ||
                                   msg.distance[pid] > point.data[2] * 100.0)
                                {
					#if defined (PRINT_CRITICAL)
					printf("msg.distance[%i]=%f, point.data[2] * 100.0=%f\n", pid, msg.distance[pid], point.data[2] * 100.0);
					#endif

			    		// added to make the result always deterministic and independent from the point order
			   	 	// in case two points get the same distance, take the one with high intensity
			    		if (((msg.distance[pid] == float(point.data[2] * 100.0)) &&  msg.intensity[pid] < float(intensity)) ||
					    (msg.distance[pid] > float(point.data[2] * 100.0)) ||
				            msg.distance[pid] == 0)
						msg.intensity[pid] = float(intensity);

					msg.distance[pid] = float(point.data[2] * 100);

                                        msg.max_y = py > msg.max_y ? py : msg.max_y;
                                        msg.min_y = py < msg.min_y ? py : msg.min_y;

					#if defined (PRINT_CRITICAL)
					printf("msg.distance[%i]=%f, msg.intensity[pid]=%f\n", pid, msg.distance[pid], msg.intensity[pid]);
					printf("msg.max_y=%i, msg.min_y=%i\n", msg.max_y, msg.min_y);
					printf("\n");
					#endif

                                }
                                if (0 == y && pointcloud2.height == 2)//process simultaneously min and max during the first layer
                                {
                                        float* fp2 = (float *)(cp + (x + (y+1)*pointcloud2.width) * pointcloud2.point_step);
                                        msg.min_height[pid] = fp[2];
                                        msg.max_height[pid] = fp2[2];
                                }
                                else
                                {
                                        msg.min_height[pid] = -1.25;
                                        msg.max_height[pid] = 0;
                                }
                        }
               }
        }
        return msg;
}

// ---------------------------------------------------------------
#if defined (OPENCL)
	#include "ocl_header.h"
	#include "stringify.h"

extern OCL_Struct OCL_objs;

	#include <array>	// std:: array
	#include <string>       // std::string
	#include <sstream>      // std::stringstream
	#include <utility>      // std::pair

	#define MAX_NUM_WORKITEMS 32
#endif
// ---------------------------------------------------------------
void points2image::run(int p) {

	pause_func();

#if defined (OPENCL)
	// constructing the OpenCL program for the points2image function
	cl_int err;
	cl_program points2image_program = clCreateProgramWithSource(OCL_objs.rcar_context, 1, (const char **)&points2image_ocl_krnl, NULL, &err);

	// building the OpenCL program for all the objects
	err = clBuildProgram(points2image_program, 1, &OCL_objs.cvengine_device, NULL, NULL, NULL);

	// kernel
	cl_kernel points2image_kernel = clCreateKernel(points2image_program, "pointcloud2_to_image", &err);

	// getting max workgroup size
	size_t local_size;
	err = clGetDeviceInfo(OCL_objs.cvengine_device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &local_size, NULL);
		//std::cout << "local_size :" << local_size << std::endl;

	// min total number of threads for executing the kernel
	cl_uint compute_unit;
	err =  clGetDeviceInfo(OCL_objs.cvengine_device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &compute_unit, NULL);

	size_t global_size = compute_unit * local_size;
		//std::cout << "global_size: " << global_size << std::endl;
#endif

	while (read_testcases < testcases)
	{
		int count = read_next_testcases(p);

		#if defined (PRINTINFO)
	      	std::cout << "# read_testcases: " << read_testcases << "  count: " << count << std::endl;
		#endif

		unpause_func();

		// ---------------------------------------------------------------
		// Expensive call ...
#if !defined (OPENCL)

		for (int i = 0; i < count; i++)
		{
			// actual kernel invocation
			results[i] = pointcloud2_to_image(pointcloud2[i], cameraExtrinsicMat[i], cameraMat[i], distCoeff[i], imageSize[i]);

			/*
			printf("msg_max_y=%i, msg_min_y=%i, msg_image_height=%i, msg_image_width=%i\n", results[i].max_y, results[i].min_y, results[i].image_height, results[i].image_width);

			for (unsigned int p=0;p<results[i].image_height*results[i].image_width;p++){
				printf("p=%u, msg_distance=%f, msg_intensity=%f, msg_min_height=%f, msg_max_height=%f\n", p, results[i].distance[p], results[i].intensity[p], results[i].min_height[p], results[i].max_height[p]);
			}
			*/
		}

		// should be replaced by OpenCL NDRange
#else
		// Set kernel parameters & launch NDRange kernel
		for (int i = 0; i < count; i++)
		{
			// Prepare inputs buffers
			size_t pc2data_numelements = pointcloud2[i].height * pointcloud2[i].width * pointcloud2[i].point_step;
			size_t size_pc2data = pc2data_numelements * sizeof(float);
				//std::cout << "pc2data_numelements: " << pc2data_numelements << std::endl;

				// Creating zero-copy buffer for pointcloud data using "CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR"
			cl_mem buff_pointcloud2_data =  clCreateBuffer(OCL_objs.rcar_context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, size_pc2data, NULL, &err);

	                        // Enqueuing mapbuffer to put the input data buff_pointcloud2_data on the map region between host and device
                        float* tmp_pointcloud2_data = (float*) clEnqueueMapBuffer(OCL_objs.cvengine_command_queue,
										  buff_pointcloud2_data, CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, 0,
										  size_pc2data, 0, 0, NULL, &err);

				// Copying from host memory to pinned host memory which is used by the CVengine automatically
			for (uint j=0; j<pc2data_numelements; j++) {
				tmp_pointcloud2_data[j] = pointcloud2[i].data[j];
			}

				// Unmapping the pointer, this will return the control to the device
                        clEnqueueUnmapMemObject(OCL_objs.cvengine_command_queue, buff_pointcloud2_data, tmp_pointcloud2_data, 0, NULL, NULL);

                        // Prepare outputs buffers
                        size_t outbuff_numelements = imageSize[i].height*imageSize[i].width;
                        size_t size_outputbuff = outbuff_numelements * sizeof(float);
                                //std::cout << "outbuff_numelements: " << outbuff_numelements << std::endl;

                        	// Allocate space in host to store results comming from GPU
	                        // These will be freed in read_next_testcases()
                        results[i].intensity  = /*new float[outbuff_numelements];*/ assign(outbuff_numelements, 0);
                        results[i].distance   = /*new float[outbuff_numelements];*/ assign(outbuff_numelements, 0);
                        results[i].min_height = /*new float[outbuff_numelements];*/ assign(outbuff_numelements, 0);
                        results[i].max_height = /*new float[outbuff_numelements];*/ assign(outbuff_numelements, 0);

                                // Creating zero-copy buffers for pids data using "CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR"
                                // These kernel args are written (not read) by the device
			cl_mem buff_pids        = clCreateBuffer(OCL_objs.rcar_context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, pointcloud2[i].width * sizeof(int),   NULL, &err);
			cl_mem buff_enable_pids = clCreateBuffer(OCL_objs.rcar_context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, pointcloud2[i].width * sizeof(int),   NULL, &err);
			cl_mem buff_pointdata2  = clCreateBuffer(OCL_objs.rcar_context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, pointcloud2[i].width * sizeof(float), NULL, &err);
			cl_mem buff_intensity   = clCreateBuffer(OCL_objs.rcar_context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, pointcloud2[i].width * sizeof(float), NULL, &err);
			cl_mem buff_py          = clCreateBuffer(OCL_objs.rcar_context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, pointcloud2[i].width * sizeof(int),   NULL, &err);
			cl_mem buff_fp_2        = clCreateBuffer(OCL_objs.rcar_context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, pointcloud2[i].width * sizeof(float), NULL, &err);

			// Set kernel parameters
			err = clSetKernelArg (points2image_kernel, 0, sizeof(int),       &pointcloud2[i].height);
			err = clSetKernelArg (points2image_kernel, 1, sizeof(int),       &pointcloud2[i].width);
 			err = clSetKernelArg (points2image_kernel, 2, sizeof(int),       &pointcloud2[i].point_step);
			err = clSetKernelArg (points2image_kernel, 3, sizeof(cl_mem),    &buff_pointcloud2_data);

				// Convert explicitly from double-precision into single-precision floating point
				// Because CVEngine supports only single precision
			SP_Mat44 tmp_cameraExtrinsic;
			SP_Mat33 tmp_cameraMat;
			SP_Vec5  tmp_distCoeff;

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

			err = clSetKernelArg (points2image_kernel, 4,  sizeof(SP_Mat44),  &tmp_cameraExtrinsic);
                       	err = clSetKernelArg (points2image_kernel, 5,  sizeof(SP_Mat33),  &tmp_cameraMat);
                        err = clSetKernelArg (points2image_kernel, 6,  sizeof(SP_Vec5),   &tmp_distCoeff);

                       	err = clSetKernelArg (points2image_kernel, 7,  sizeof(ImageSize), &imageSize[i]);
                        err = clSetKernelArg (points2image_kernel, 8,  sizeof(cl_mem), &buff_pids);
                        err = clSetKernelArg (points2image_kernel, 9,  sizeof(cl_mem), &buff_enable_pids);
			err = clSetKernelArg (points2image_kernel, 10, sizeof(cl_mem), &buff_pointdata2);
			err = clSetKernelArg (points2image_kernel, 11, sizeof(cl_mem), &buff_intensity);
			err = clSetKernelArg (points2image_kernel, 12, sizeof(cl_mem), &buff_py);
			err = clSetKernelArg (points2image_kernel, 13, sizeof(cl_mem), &buff_fp_2);

			// Update global size
                        size_t tmp_size =  pointcloud2[i].width / /*MAX_NUM_WORKITEMS*/ local_size;
                                //std::cout << "# work-groups : " << tmp_size << std::endl;

                        global_size = (tmp_size + 1) * /*MAX_NUM_WORKITEMS*/ local_size; // ~ 50000
                                //std::cout << "global_size : " << global_size << std::endl;

			// Launch kernel on device
                        err = clEnqueueNDRangeKernel(OCL_objs.cvengine_command_queue, points2image_kernel, 1, NULL,  &global_size, &local_size, 0, NULL, NULL);

	                // CPU update of msg_intensity, msg_distance, msg_min_height, msg_max_height, etc
                        size_t nelems_tmp     = pointcloud2[i].width;
                        size_t size_tmp_int   = nelems_tmp * sizeof(int);
                        size_t size_tmp_float = nelems_tmp * sizeof(float);

/*
                        std::vector<int>    cpu_pids        (nelems_tmp);
                        std::vector<int>    cpu_enable_pids (nelems_tmp);
                        std::vector<float>  cpu_pointdata2  (nelems_tmp);
                        std::vector<float>  cpu_intensity   (nelems_tmp);
                        std::vector<int>    cpu_py          (nelems_tmp);
                        std::vector<float>  cpu_fp_2        (nelems_tmp);
*/

/*
                        int* tmpmap_pids = (int*) clEnqueueMapBuffer(OCL_objs.cvengine_command_queue, buff_pids, CL_TRUE, CL_MAP_READ, 0, size_tmp_int, 0, 0, NULL, &err);
			for (uint p=0; p<nelems_tmp; p++) {
                        	cpu_pids[p] = tmpmap_pids[p];
			}
                        clEnqueueUnmapMemObject(OCL_objs.cvengine_command_queue, buff_pids, tmpmap_pids, 0, NULL, NULL);
*/
                       	int* cpu_pids = (int*) clEnqueueMapBuffer(OCL_objs.cvengine_command_queue, buff_pids, CL_TRUE, CL_MAP_READ, 0, size_tmp_int, 0, 0, NULL, &err);

/*
                        int* tmpmap_enable_pids = (int*) clEnqueueMapBuffer(OCL_objs.cvengine_command_queue, buff_enable_pids, CL_TRUE, CL_MAP_READ, 0, size_tmp_int, 0, 0, NULL, &err);
			for (uint p=0; p<nelems_tmp; p++) {
                        	cpu_enable_pids[p] = tmpmap_enable_pids[p];
			}
                        clEnqueueUnmapMemObject(OCL_objs.cvengine_command_queue, buff_enable_pids, tmpmap_enable_pids, 0, NULL, NULL);
*/
                        int* cpu_enable_pids = (int*) clEnqueueMapBuffer(OCL_objs.cvengine_command_queue, buff_enable_pids, CL_TRUE, CL_MAP_READ, 0, size_tmp_int, 0, 0, NULL, &err);

/*
                        float* tmpmap_pointdata2 = (float*) clEnqueueMapBuffer(OCL_objs.cvengine_command_queue, buff_pointdata2, CL_TRUE, CL_MAP_READ, 0,size_tmp_float, 0, 0, NULL, &err);
			for (uint p=0; p<nelems_tmp; p++) {
                        	cpu_pointdata2[p] = tmpmap_pointdata2[p];
			}
                        clEnqueueUnmapMemObject(OCL_objs.cvengine_command_queue, buff_pointdata2, tmpmap_pointdata2, 0, NULL, NULL);
*/

                        float* cpu_pointdata2 = (float*) clEnqueueMapBuffer(OCL_objs.cvengine_command_queue, buff_pointdata2, CL_TRUE, CL_MAP_READ, 0,size_tmp_float, 0, 0, NULL, &err);

/*
                        float* tmpmap_intensity = (float*) clEnqueueMapBuffer(OCL_objs.cvengine_command_queue, buff_intensity, CL_TRUE, CL_MAP_READ, 0, size_tmp_float, 0, 0, NULL, &err);
			for (uint p=0; p<nelems_tmp; p++) {
	                        cpu_intensity[p] = tmpmap_intensity[p];
			}
                        clEnqueueUnmapMemObject(OCL_objs.cvengine_command_queue, buff_intensity, tmpmap_intensity, 0, NULL, NULL);
*/
                        float* cpu_intensity = (float*) clEnqueueMapBuffer(OCL_objs.cvengine_command_queue, buff_intensity, CL_TRUE, CL_MAP_READ, 0, size_tmp_float, 0, 0, NULL, &err);

/*
                        int* tmpmap_py = (int*) clEnqueueMapBuffer(OCL_objs.cvengine_command_queue, buff_py, CL_TRUE, CL_MAP_READ, 0, size_tmp_int, 0, 0, NULL, &err);
			for (uint p=0; p<nelems_tmp; p++) {
	                        cpu_py[p] = tmpmap_py[p];
			}
                        clEnqueueUnmapMemObject(OCL_objs.cvengine_command_queue, buff_py, tmpmap_py, 0, NULL, NULL);
*/
                        int* cpu_py = (int*) clEnqueueMapBuffer(OCL_objs.cvengine_command_queue, buff_py, CL_TRUE, CL_MAP_READ, 0, size_tmp_int, 0, 0, NULL, &err);

/*
                        int* tmpmap_fp_2 = (int*) clEnqueueMapBuffer(OCL_objs.cvengine_command_queue, buff_fp_2, CL_TRUE, CL_MAP_READ, 0, size_tmp_float, 0, 0, NULL, &err);
			for (uint p=0; p<nelems_tmp; p++) {
	                        cpu_fp_2[p] = tmpmap_fp_2[p];
			}
                        clEnqueueUnmapMemObject(OCL_objs.cvengine_command_queue, buff_fp_2, tmpmap_fp_2, 0, NULL, NULL);
*/
                        int* cpu_fp_2 = (int*) clEnqueueMapBuffer(OCL_objs.cvengine_command_queue, buff_fp_2, CL_TRUE, CL_MAP_READ, 0, size_tmp_float, 0, 0, NULL, &err);
 
			// Get result back to host
			/*
			printf("msg_max_y=%i, msg_min_y=%i, msg_image_height=%i, msg_image_width=%i\n", results[i].max_y, results[i].min_y, results[i].image_height, results[i].image_width);

			for (unsigned int p=0;p<results[i].image_height*results[i].image_width;p++){
				printf("p=%u, msg_distance=%f, msg_intensity=%f, msg_min_height=%f, msg_max_height=%f\n", p, results[i].distance[p], results[i].intensity[p], results[i].min_height[p], results[i].max_height[p]);
			}
			*/

				// Getting width and heights
			//const int w          = imageSize[i].width;
			const int h          = imageSize[i].height;
			const int pc2_height = pointcloud2[i].height;
			const int pc2_width  = pointcloud2[i].width;
			const int pc2_pstep  = pointcloud2[i].point_step;
			/*
			std::cout << "w: " << imageSize[i].width  << std::endl;
			std::cout << "h: " << imageSize[i].height << std::endl;
			std::cout << "pointcloud2[" << i << "].height: "     << pointcloud2[i].height     << std::endl;
			std::cout << "pointcloud2[" << i << "].width: "      << pointcloud2[i].width      << std::endl;
			std::cout << "pointcloud2[" << i << "].point_step: " << pointcloud2[i].point_step << std::endl;
			*/

			// Writing msg scalars to global memory
			results[i].max_y        = -1;
			results[i].min_y        = h;
			results[i].image_height = imageSize[i].height;
			results[i].image_width  = imageSize[i].width;

			// Defining a global const pointer
			uintptr_t cp = (uintptr_t)pointcloud2[i].data;
			//__global const float* cp = (__global const float *)(pointcloud2_data);

			// From now on, we executed in serially and in order
			for (unsigned int y = 0; y < pc2_height; ++y) {
				for (unsigned int x = 0; x < pc2_width; x++) {

					if (cpu_enable_pids[x] == 1) {
						//int pid = py * w + px;
						int pid = cpu_pids [x];
						/*double*/ float tmp_pointdata2 = cpu_pointdata2[x] * 100;

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

						// Process simultaneously min and max during the first layer
						if (0 == y && pc2_height == 2) {
							//__global const float* fp2 = (__global const float *)(cp + (x + (y+1)*pc2_width) * pc2_pstep);
							float* fp2 = (float *)(cp + (x + (y+1)*pointcloud2[i].width) * pointcloud2[i].point_step);
							results[i].min_height[pid] = /*fp[2]*/ cpu_fp_2[x];
							results[i].max_height[pid] = fp2[2];
						}
						else {
							results[i].min_height[pid] = -1.25f;
							results[i].max_height[pid] = 0.0f;
						}
					} // End: if (cpu_enable_pids[x] == 1) {
			       } // End: for (unsigned int x = 0; x < pc2_width; x++) {
			} // End: for (unsigned int y = 0; y < pc2_height; ++y) {

			clEnqueueUnmapMemObject(OCL_objs.cvengine_command_queue, buff_pids, cpu_pids, 0, NULL, NULL);
			clEnqueueUnmapMemObject(OCL_objs.cvengine_command_queue, buff_enable_pids, cpu_enable_pids, 0, NULL, NULL);
			clEnqueueUnmapMemObject(OCL_objs.cvengine_command_queue, buff_pointdata2, cpu_pointdata2, 0, NULL, NULL);
			clEnqueueUnmapMemObject(OCL_objs.cvengine_command_queue, buff_intensity, cpu_intensity, 0, NULL, NULL);
			clEnqueueUnmapMemObject(OCL_objs.cvengine_command_queue, buff_py, cpu_py, 0, NULL, NULL);
			clEnqueueUnmapMemObject(OCL_objs.cvengine_command_queue, buff_fp_2, cpu_fp_2, 0, NULL, NULL);

			clReleaseMemObject(buff_pointcloud2_data);
			clReleaseMemObject(buff_pids);
			clReleaseMemObject(buff_enable_pids);
			clReleaseMemObject(buff_pointdata2);
			clReleaseMemObject(buff_intensity);
			clReleaseMemObject(buff_py);
			clReleaseMemObject(buff_fp_2);
		}
#endif
		// End of OpenCL NDRange
		// ---------------------------------------------------------------

		/*
		#if defined (PRINTINFO)
		std::cout << "Outputs will be checked ... " << std::endl;
		#endif
		*/

	      	pause_func();
	      	check_next_outputs(count);
    	}

#if defined (OPENCL)
	err = clReleaseKernel(points2image_kernel);
	err = clReleaseProgram(points2image_program);
#endif

}

void points2image::check_next_outputs(int count)
{
  PointsImage reference;

  for (int i = 0; i < count; i++)
    {
      parsePointsImage(output_file, &reference);
      if ((results[i].image_height != reference.image_height)
	  || (results[i].image_width != reference.image_width))
      {
	  error_so_far = true;

          #if defined (PRINTINFO)
          std::cout << "    image_height: " << results[i].image_height << std::endl;
          std::cout << "    image_width: "  << results[i].image_width  << std::endl;
          #endif
      }
      if ((results[i].min_y != reference.min_y)
	  || (results[i].max_y != reference.max_y))
      {
	  error_so_far = true;

          #if defined (PRINTINFO)
          std::cout << "   min_y: " << results[i].min_y << std::endl;
          std::cout << "   max_y: " << results[i].max_y << std::endl;
          #endif
      }

      int pos = 0;
      for (int h = 0; h < reference.image_height; h++)
	for (int w = 0; w < reference.image_width; w++)
	  {
	    if (fabs(reference.intensity[pos] - results[i].intensity[pos]) > max_delta) {
	      max_delta = fabs(reference.intensity[pos] - results[i].intensity[pos]);

	      #if defined (PRINTINFO)
	      std::cout << "	intensity: " << " h: " << h  << " w: " << w << " max_delta: " << max_delta << " pos: " << pos << std::endl;
	      #endif
	    }

	    if (fabs(reference.distance[pos] - results[i].distance[pos]) > max_delta) {
	      max_delta = fabs(reference.distance[pos] - results[i].distance[pos]);

	      #if defined (PRINTINFO)
	      std::cout << "	distance: " << " h: " << h  << " w: " << w << " max_delta: " << max_delta << " pos: " << pos << std::endl;
	      #endif
	    }

	    if (fabs(reference.min_height[pos] - results[i].min_height[pos]) > max_delta) {
	      max_delta = fabs(reference.min_height[pos] - results[i].min_height[pos]);

	      #if defined (PRINTINFO)
	      std::cout << "	min_height: " << " h: " << h  << " w: " << w << " max_delta: " << max_delta << " pos: " << pos << std::endl;
	      #endif
    	    }

	    if (fabs(reference.max_height[pos] - results[i].max_height[pos]) > max_delta) {
	      max_delta = fabs(reference.max_height[pos] - results[i].max_height[pos]);

	      #if defined (PRINTINFO)
	      std::cout << "	max_height: " << " h: " << h  << " w: " << w << " max_delta: " << max_delta << " pos: " << pos << std::endl;
	      #endif
            }

	    pos++;
	  }

      delete [] reference.intensity;
      delete [] reference.distance;
      delete [] reference.min_height;
      delete [] reference.max_height;
    }
}

bool points2image::check_output() {
    std::cout << "checking output \n";

    input_file.close();
    output_file.close();

    std::cout << "max delta: " << max_delta << "\n";

    #if defined (PRINTINFO)
    std::cout << "MAX_EPS: " << MAX_EPS << std::endl;
    #endif

    if ((max_delta > MAX_EPS) || error_so_far)
	  return false;
    return true;
}

points2image a = points2image();
kernel& myKernel = a;
