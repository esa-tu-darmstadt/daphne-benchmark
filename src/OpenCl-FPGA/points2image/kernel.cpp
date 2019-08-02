/**
 * Author:  Florian Stock, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2018
 * Author:  Leonardo Solis, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2018
 * License: Apache 2.0 (see attachached File)
 */
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
  PointCloud2* pointcloud2 = NULL;
  Mat44* cameraExtrinsicMat = NULL;
  Mat33* cameraMat = NULL;
  Vec5* distCoeff = NULL;
  ImageSize* imageSize = NULL;
  PointsImage* results = NULL;
protected:
  virtual int read_next_testcases(int count);
  virtual void check_next_outputs(int count);
  int read_testcases = 0;
  std::ifstream input_file, output_file;
  bool error_so_far;
  double max_delta;
};

void  parsePointCloud(std::ifstream& input_file, PointCloud2* pointcloud2) {
    try {
	input_file.read((char*)&(pointcloud2->height), sizeof(int32_t));
	input_file.read((char*)&(pointcloud2->width), sizeof(int32_t));
	input_file.read((char*)&(pointcloud2->point_step), sizeof(uint32_t));
	int pos = 0;
	pointcloud2->data = new float[pointcloud2->height * pointcloud2->width * pointcloud2->point_step];
	input_file.read((char*)pointcloud2->data, pointcloud2->height * pointcloud2->width * pointcloud2->point_step);
    }  catch (std::ifstream::failure e) {
	std::cerr << "Error reading file\n";
	exit(-3);
    }
}

void  parseCameraExtrinsicMat(std::ifstream& input_file, Mat44* cameraExtrinsicMat) {
    double temp;
    try {
	for (int h = 0; h < 4; h++)
	    for (int w = 0; w < 4; w++) {
                input_file.read((char*)&temp, sizeof(double));
		cameraExtrinsicMat->data[h][w] = temp;
            }
    } catch (std::ifstream::failure e) {
	std::cerr << "Error reading file\n";
	exit(-3);
    }
}

  
void parseCameraMat(std::ifstream& input_file, Mat33* cameraMat ) {
    double temp;
    try {
	for (int h = 0; h < 3; h++)
	    for (int w = 0; w < 3; w++) {
		input_file.read((char*)&(temp), sizeof(double));
		cameraMat->data[h][w] = temp;
            }
    } catch (std::ifstream::failure e) {
	std::cerr << "Error reading file\n";
	exit(-3);
    }
}
  
void  parseDistCoeff(std::ifstream& input_file, Vec5* distCoeff) {
    double temp;
    try {
	for (int w = 0; w < 5; w++) {
	    input_file.read((char*)&(temp), sizeof(double));
	    distCoeff->data[w] = temp;
        }
    } catch (std::ifstream::failure e) {
	std::cerr << "Error reading file\n";
	exit(-3);
    }
}
  
void  parseImageSize(std::ifstream& input_file, ImageSize* imageSize) {
    try {
	input_file.read((char*)&(imageSize->width), sizeof(int32_t));
	input_file.read((char*)&(imageSize->height), sizeof(int32_t));
    } catch (std::ifstream::failure e) {
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
    } catch (std::ifstream::failure e) {
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
    #if defined (SW_EMU)
    testcases = 10;
    #else
    testcases = /*2500*/2500;
    #endif

    input_file.exceptions ( std::ifstream::failbit | std::ifstream::badbit );
    output_file.exceptions ( std::ifstream::failbit | std::ifstream::badbit );
    try {
        #if defined (SW_EMU)
	input_file.open("input_10testcases.dat", std::ios::binary);
	output_file.open("output_singleprec_10testcases.dat", std::ios::binary);
	#else
	input_file.open("/run/media/mmcblk0p2/data/input.dat", std::ios::binary);
	//output_file.open("/run/media/mmcblk0p2/data/output.dat", std::ios::binary);
	output_file.open("/run/media/mmcblk0p2/data/output_singleprec.dat", std::ios::binary);
	//output_file.open("/run/media/mmcblk0p2/data/output_singleprec_10testcases.dat", std::ios::binary);
	#endif
    } catch (std::ifstream::failure e) {
	std::cerr << "Error opening file\n";
	exit(-2);
    }
    error_so_far = false;
    max_delta = 0.0;


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

#if defined (XILINX_OPENCL_EPHOS)
#include "ocl_ephos.h"
#endif

/**
   This code is extracted from Autoware, file:
   ~/Autoware/ros/src/sensing/fusion/packages/points2image/lib/points_image/points_image.cpp
*/

PointsImage
pointcloud2_to_image(const PointCloud2& pointcloud2,
                     const Mat44& cameraExtrinsicMat,
                     const Mat33& cameraMat, const Vec5& distCoeff,
                     const ImageSize& imageSize

                     #if defined (XILINX_OPENCL_EPHOS)
                     ,           
	             Struct_OCLEphos* obj_OCLEphos
                     #endif
                    )
{
    int w = imageSize.width;
    int h = imageSize.height;
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
    msg.max_y = -1;
    msg.min_y = h;
        
    msg.image_height = imageSize.height;
    msg.image_width = imageSize.width;

    #if defined (XILINX_OPENCL_EPHOS)

    //Copying struct values to local vars (for cleaner code)
    cl::Context context = obj_OCLEphos->context;
    cl::CommandQueue q  = obj_OCLEphos->q;
    cl::Kernel kernel   = obj_OCLEphos->kernel;

    //Creating Input Buffers inside Device
    size_t pc2data_nelems = pointcloud2.height * pointcloud2.width * pointcloud2.point_step;
    size_t pc2data_nbytes = pc2data_nelems * sizeof(float);
    cl::Buffer buffer_pointcloud2_data(context, CL_MEM_READ_ONLY, pc2data_nbytes);

    //Creating Output Buffers inside Device
    //These are considered CL_MEM_READ_WRITE buffers
    // as they are initialized in host: see assign(w*h, 0) above
    size_t msg_nelems = w * h;
    size_t msg_nbytes = msg_nelems * sizeof(float);
    cl::Buffer buffer_msg_intensity  (context, CL_MEM_READ_WRITE, msg_nbytes);
    cl::Buffer buffer_msg_distance   (context, CL_MEM_READ_WRITE, msg_nbytes);
    cl::Buffer buffer_msg_min_height (context, CL_MEM_READ_WRITE, msg_nbytes);
    cl::Buffer buffer_msg_max_height (context, CL_MEM_READ_WRITE, msg_nbytes);
    cl::Buffer buffer_msg_max_y_min_y(context, CL_MEM_WRITE_ONLY, 8); // 2 ints

    //Copying input data to Device buffer from host memory
    q.enqueueWriteBuffer(buffer_pointcloud2_data, CL_TRUE, 0, pc2data_nbytes, pointcloud2.data);

    q.enqueueWriteBuffer(buffer_msg_intensity,  CL_TRUE, 0, msg_nbytes, msg.intensity);
    q.enqueueWriteBuffer(buffer_msg_distance,   CL_TRUE, 0, msg_nbytes, msg.distance);
    q.enqueueWriteBuffer(buffer_msg_min_height, CL_TRUE, 0, msg_nbytes, msg.min_height);
    q.enqueueWriteBuffer(buffer_msg_max_height, CL_TRUE, 0, msg_nbytes, msg.max_height);

    //Setting kernel args
    unsigned int nargs = 0;
    unsigned int width_times_step = pointcloud2.width * pointcloud2.point_step;

    kernel.setArg(nargs++, buffer_pointcloud2_data);
    kernel.setArg(nargs++, buffer_msg_intensity);
    kernel.setArg(nargs++, buffer_msg_distance);
    kernel.setArg(nargs++, buffer_msg_min_height);
    kernel.setArg(nargs++, buffer_msg_max_height);
    kernel.setArg(nargs++, buffer_msg_max_y_min_y);
    kernel.setArg(nargs++, pointcloud2.height);
    kernel.setArg(nargs++, pointcloud2.width);
    kernel.setArg(nargs++, pointcloud2.point_step);
    kernel.setArg(nargs++, width_times_step);
    kernel.setArg(nargs++, invT);
    kernel.setArg(nargs++, invR);
    kernel.setArg(nargs++, distCoeff);
    kernel.setArg(nargs++, cameraMat);
    kernel.setArg(nargs++, w);
    kernel.setArg(nargs++, h);
    kernel.setArg(nargs++, msg.max_y);
    kernel.setArg(nargs++, msg.min_y);

    //Running Kernel
    q.enqueueTask(kernel);

    //q.finish();

    //Copying Device result data to Host memory
    int temp[2];
    q.enqueueReadBuffer(buffer_msg_intensity,   CL_TRUE, 0, msg_nbytes, msg.intensity);
    q.enqueueReadBuffer(buffer_msg_distance,    CL_TRUE, 0, msg_nbytes, msg.distance);
    q.enqueueReadBuffer(buffer_msg_min_height,  CL_TRUE, 0, msg_nbytes, msg.min_height);
    q.enqueueReadBuffer(buffer_msg_max_height,  CL_TRUE, 0, msg_nbytes, msg.max_height);
    q.enqueueReadBuffer(buffer_msg_max_y_min_y, CL_TRUE, 0, 8, &temp);

    q.finish();

    msg.max_y = temp[0];
    msg.min_y = temp[1];

    #else
    for (uint32_t y = 0; y < pointcloud2.height; ++y) {
	for (uint32_t x = 0; x < pointcloud2.width; ++x) {
	    float* fp = (float *)(cp + (x + y*pointcloud2.width) * pointcloud2.point_step);
	    float intensity = fp[4];

	    Mat13 point, point2;
	    point2.data[0] = float(fp[0]);
	    point2.data[1] = float(fp[1]);
	    point2.data[2] = float(fp[2]);
	    //point = point * invR.t() + invT.t();
	    for (int row = 0; row < 3; row++) {
		point.data[row] = invT.data[row];
		for (int col = 0; col < 3; col++) 
		    point.data[row] += point2.data[col] * invR.data[row][col];
	    }
                        
	    if (point.data[2] <= 2.5) {
		continue;
	    }

	    float tmpx = point.data[0] / point.data[2];
	    float tmpy = point.data[1]/ point.data[2];
	    float r2 = tmpx * tmpx + tmpy * tmpy;
	    float tmpdist = 1 + distCoeff.data[0] * r2
		+ distCoeff.data[1] * r2 * r2
		+ distCoeff.data[4] * r2 * r2 * r2;

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
	    if(0 <= px && px < w && 0 <= py && py < h)
		{
		    int pid = py * w + px;
		    if(msg.distance[pid] == 0 ||
		       msg.distance[pid] >= float(point.data[2] * 100.0))
			{
			    // added to make the result always deterministic and independent from the point order
			    // in case two points get the same distance, take the one with high intensity
			    if (((msg.distance[pid] == float(point.data[2] * 100.0)) &&  msg.intensity[pid] < float(intensity)) ||
				(msg.distance[pid] > float(point.data[2] * 100.0)) ||
				msg.distance[pid] == 0)
				msg.intensity[pid] = float(intensity);
				    
			    msg.distance[pid] = float(point.data[2] * 100.0);
				    
			    msg.max_y = py > msg.max_y ? py : msg.max_y;
			    msg.min_y = py < msg.min_y ? py : msg.min_y;
				    
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
    #endif // End of #if defined (XILINX_OPENCL_EPHOS)
    return msg;
}

void points2image::run(int p) {
    pause_func();

    #if defined (XILINX_OPENCL_EPHOS)

    //Getting Xilinx Platform and its device
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];
    std::string device_name = device.getInfo<CL_DEVICE_NAME>();

    #if defined (PRINTINFO)
    std::cout << "EPHoS FPGA device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
    #endif

    //Creating Context and Command Queue for selected Device
    cl::Context context(device);
    cl::CommandQueue q(context, device);

    //Loading XCL Bin into char buffer
    std::string binaryFile = xcl::find_binary_file(device_name, "pointcloud2_to_image");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    devices.resize(1);
    cl::Program program(context, devices, bins);

    //Creating Kernel and Functor of Kernel
    int err1;
    cl::Kernel kernel(program, "pointcloud2_to_image", &err1);

    //Copying values to global-scope struct
    Struct_OCLEphos obj_OCLEphos;
    obj_OCLEphos.context = context;
    obj_OCLEphos.q       = q;
    obj_OCLEphos.kernel  = kernel;

    #endif // End of #if defined (XILINX_OPENCL_EPHOS)
  
    while (read_testcases < testcases)
	{
	    int count = read_next_testcases(p);

	    #if defined (PRINTINFO)
	    std::cout << "# read_testcases: " << read_testcases << "  count: " << count << std::endl;
            #endif

	    unpause_func();
	    for (int i = 0; i < count; i++)
		{
		    // actual kernel invocation
		    results[i] = pointcloud2_to_image(pointcloud2[i],
						      cameraExtrinsicMat[i],
						      cameraMat[i], distCoeff[i],
						      imageSize[i]
                                                      #if defined (XILINX_OPENCL_EPHOS)
                                                      ,
						      &obj_OCLEphos
						      #endif
						      );
		}
	    pause_func();
	    check_next_outputs(count);
	}

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
    if ((max_delta > MAX_EPS) || error_so_far)
	return false;
    return true;
}

points2image a = points2image();
kernel& myKernel = a;
