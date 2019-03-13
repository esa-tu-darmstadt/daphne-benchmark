#include "benchmark.h"
#include "datatypes.h"
#include <math.h>
#include <iostream>
#include <fstream>

#define MAX_EPS 0.001
#define THREADS 256

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

__device__ __managed__ float result_buffer[800*600*4];

  void  parsePointCloud(std::ifstream& input_file, PointCloud2* pointcloud2) {
    input_file.read((char*)&(pointcloud2->height), sizeof(int32_t));
    input_file.read((char*)&(pointcloud2->width), sizeof(int32_t));
    input_file.read((char*)&(pointcloud2->point_step), sizeof(uint32_t));
    cudaMallocManaged(&pointcloud2->data, pointcloud2->height * pointcloud2->width * pointcloud2->point_step);
    input_file.read((char*)pointcloud2->data, pointcloud2->height * pointcloud2->width * pointcloud2->point_step);
  }

  void  parseCameraExtrinsicMat(std::ifstream& input_file, Mat44* cameraExtrinsicMat) {
    for (int h = 0; h < 4; h++)
      for (int w = 0; w < 4; w++)
        input_file.read((char*)&(cameraExtrinsicMat->data[h][w]),sizeof(double));
    //input_file.read((char*)&(cameraExtrinsicMat->data[0][0]), 16*sizeof(double));
  }

  
  void parseCameraMat(std::ifstream& input_file, Mat33* cameraMat ) {
    for (int h = 0; h < 3; h++)
      for (int w = 0; w < 3; w++)
        input_file.read((char*)&(cameraMat->data[h][w]), sizeof(double));
        //    input_file.read((char*)&(cameraMat->data[0][0]), 9*sizeof(double));
  }
  
  void  parseDistCoeff(std::ifstream& input_file, Vec5* distCoeff) {
      for (int w = 0; w < 5; w++)
        input_file.read((char*)&(distCoeff->data[w]), sizeof(double));
  }
  
  void  parseImageSize(std::ifstream& input_file, ImageSize* imageSize) {
    input_file.read((char*)&(imageSize->width), sizeof(int32_t));
    input_file.read((char*)&(imageSize->height), sizeof(int32_t));
  }

void parsePointsImage(std::ifstream& output_file, PointsImage* goldenResult) {  
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

// return how many could be read
int points2image::read_next_testcases(int count)
{
  int i;
  
  if (pointcloud2) 
    for (int m = 0; m < count; ++m)
      cudaFree(pointcloud2[m].data);     
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
        cudaFree(results[m].intensity);
	//        cudaFree(results[m].distance);
	//cudaFree(results[m].min_height);
	//cudaFree(results[m].max_height);
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
  testcases = 25; //2500;
  
  input_file.open("../../../data/p2i_input.dat", std::ios::binary);
  output_file.open("../../../data/p2i_output.dat", std::ios::binary);
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

  cudaMallocManaged(&result, sizeof(float) * count);
  //cudaMemset(result, int(value), sizeof(float) * count);
  //  for (int i = 0; i < count; i++) {
  //  result[i] = value;
  //}
  return result;
}

/**
   Cuda has no atomicMin for floats, so improvise one.
   As we use it for distance, we consider just the >=0 case
*/
__device__ __forceinline__ float atomicFloatMin(float * addr, float value) {	
 return  __int_as_float(atomicMin((int *)addr, __float_as_int(value)));

}

/** 
 Computation on the GPU
*/
__global__ void compute_point_from_pointcloud(const float*  __restrict__ cp, float*  volatile msg_distance, float* volatile msg_intensity,
					      float * __restrict__ msg_min_height, int width, int height, int point_step,
					      int w, int h, Mat33 invR, Mat13 invT, Vec5 distCoeff, Mat44 cameraExtrinsicMat, Mat33 cameraMat,
					      int* __restrict__ min_y, int* __restrict__ max_y) {
  int y = blockIdx.x; // * blockDim.x + threadIdx.x;
  int x = blockIdx.y * THREADS + threadIdx.x;
  for (int j = 0; j < 1; j++) {
  
  if (x >= width)
    continue;
  
  
  float* fp = (float *)((uintptr_t)cp + (x + y*width) * point_step);
  double intensity = fp[4];

  Mat13 point, point2;
  point2.data[0] = double(fp[0]);
  point2.data[1] = double(fp[1]);
  point2.data[2] = double(fp[2]);
  //  printf("x: %i y: %i ,  msg_distance, intensity: %f %f\n", x,y,  point2.data[2]*100.0, intensity);

  //point = point * invR.t() + invT.t();
  for (int row = 0; row < 3; row++) {
    point.data[row] = invT.data[row];
    for (int col = 0; col < 3; col++) 
      point.data[row] += point2.data[col] * invR.data[row][col];
  }

  //  printf("x: %i y: %i ,  msg_distance, intensity: %f %f\n", x,y,  point.data[2]*100.0, intensity);
  
  if (point.data[2] <= 2.5) {
    continue;
  }

  //printf("x: %i y: %i\n", x,y);
  double tmpx = point.data[0] / point.data[2];
  double tmpy = point.data[1]/ point.data[2];
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
  imagepoint.x = cameraMat.data[0][0] * imagepoint.x + cameraMat.data[0][2];
  imagepoint.y = cameraMat.data[1][1] * imagepoint.y + cameraMat.data[1][2];

  int px = int(imagepoint.x + 0.5);
  int py = int(imagepoint.y + 0.5);
  float cm_point, oldvalue;
  int pid;
  if (0 <= px && px < w && 0 <= py && py < h)
    {
      pid = py * w + px;
      cm_point = point.data[2] * 100.0;
      oldvalue = msg_distance[pid];
      atomicCAS((int*)&msg_distance[pid], 0, __float_as_int(cm_point));
      atomicFloatMin(&msg_distance[pid], cm_point);
      // in case some other thread also wrote something, or something was previous in
    }
  __syncthreads();
  __threadfence_system();
  float newvalue = msg_distance[pid];
  if (0 <= px && px < w && 0 <= py && py < h)
    {
      // if this threads value is the actual new distance, update the other values as well
      if ( newvalue>= cm_point)
	{

	  msg_intensity[pid] = float(intensity);
	  atomicMax(max_y, py);
	  atomicMin(min_y, py);
	  
	}
      msg_min_height[pid] = -1.25;
    }
  __syncthreads();
  }
}

/**
   This code is extracted from Autoware, file:
   ~/Autoware/ros/src/sensing/fusion/packages/points2image/lib/points_image/points_image.cpp
*/


PointsImage
pointcloud2_to_image(const PointCloud2& pointcloud2,
                     const Mat44& cameraExtrinsicMat,
                     const Mat33& cameraMat, const Vec5& distCoeff,
                     const ImageSize& imageSize)
{
        int w = imageSize.width;
        int h = imageSize.height;
        
        PointsImage msg;

	msg.intensity = result_buffer;
        msg.distance = msg.intensity + h*w;
        msg.min_height = msg.distance + h*w;
        msg.max_height = msg.min_height + h*w;
	for (int i = 0; i < h*w; i++) {
	  msg.intensity[i] = 0.0;
	  msg.distance[i] = 0.0;
	  msg.min_height[i] = 0.0;
	  msg.max_height[i] = 0.0;
	}

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

	// cuda result allocation
	int *cuda_min_y, *cuda_max_y;
        cudaMalloc(&cuda_min_y, sizeof(int));
        cudaMalloc(&cuda_max_y, sizeof(int));
	cudaMemcpy(cuda_min_y, &msg.min_y, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_max_y, &msg.max_y, sizeof(int), cudaMemcpyHostToDevice);

	//	std::cout << " h: " << h << " w: " << w << " pointcloud2.height, width: " << pointcloud2.height << ", " <<  pointcloud2.width << "\n";

	
	// first approach: each block consists of THREADS pixels of one line, with one thread per pixel
	dim3 threaddim(THREADS);
	dim3 blockdim(pointcloud2.height, (pointcloud2.width+THREADS-1)/THREADS); // round up, so we dont miss one
	compute_point_from_pointcloud<<<blockdim, threaddim>>>(pointcloud2.data, msg.distance,
							       msg.intensity, msg.min_height,
							       pointcloud2.width, pointcloud2.height, pointcloud2.point_step,
							       w, h,
							       invR, invT, distCoeff, cameraExtrinsicMat, cameraMat,
							       cuda_min_y, cuda_max_y);
        cudaDeviceSynchronize();
        cudaMemcpy(&msg.min_y, cuda_min_y, sizeof(int),
		   cudaMemcpyDeviceToHost);
        cudaMemcpy(&msg.max_y, cuda_max_y, sizeof(int),
		   cudaMemcpyDeviceToHost);
	cudaFree(cuda_max_y);
	cudaFree(cuda_min_y);
	//	cudaMemcpy(msg.distance, cudaDistance, result_size,
	//	   cudaMemcpyDeviceToHost);
	//cudaMemcpy(msg.intensity, cudaIntensity, result_size,
	//	   cudaMemcpyDeviceToHost);
	//cudaMemcpy(msg.min_height, cudaMin_height, result_size,
	//	   cudaMemcpyDeviceToHost);

	//cudaFree(cudaPointCloud2_data);
	//cudaFree(cudaDistance);
	//cudaFree(cudaIntensity);
	//cudaFree(cudaMin_height);
        return msg;
}


void points2image::run(int p) {
  pause_func();
  
  while (read_testcases < testcases)
    {
      int count = read_next_testcases(p);
      cudaDeviceSynchronize();
      unpause_func();
      for (int i = 0; i < count; i++)
        {
          // actual kernel invocation
          results[i] = pointcloud2_to_image(pointcloud2[i],
                                            cameraExtrinsicMat[i],
                                            cameraMat[i], distCoeff[i],
                                            imageSize[i]);
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
        }
      if ((results[i].min_y != reference.min_y)
          || (results[i].max_y != reference.max_y))
        {
          error_so_far = true;
        }
      
      int pos = 0;
      for (int h = 0; h < reference.image_height; h++)
        for (int w = 0; w < reference.image_width; w++)
          {
	    
            if (fabs(reference.intensity[pos] - results[i].intensity[pos]) > max_delta)
              max_delta = fabs(reference.intensity[pos] - results[i].intensity[pos]);
	    if (fabs(reference.distance[pos] - results[i].distance[pos]) > max_delta)
              max_delta = fabs(reference.distance[pos] - results[i].distance[pos]);
            if (fabs(reference.min_height[pos] - results[i].min_height[pos]) > max_delta)
              max_delta = fabs(reference.min_height[pos] - results[i].min_height[pos]);
            if (fabs(reference.max_height[pos] - results[i].max_height[pos]) > max_delta)
              max_delta = fabs(reference.max_height[pos] - results[i].max_height[pos]);
	    /*        if (max_delta > 0.0) {
	    std::cout << "h,w:" << h << "," << w << " actual : " << results[i].intensity[pos] << "  expected: " << reference.intensity[pos] << "\n";
	    std::cout << "actual : " << results[i].distance[pos] << "  expected: " << reference.distance[pos] << "\n";
	    
	    std::cout << "actual : " << results[i].min_height[pos] << "  expected: " << reference.min_height[pos] << "\n";
	    std::cout << "actual : " << results[i].max_height[pos] << "  expected: " << reference.max_height[pos] << "\n";
	    exit(3); 
	    } */
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
