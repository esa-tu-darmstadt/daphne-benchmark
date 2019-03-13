#include "benchmark.h"
#include "datatypes.h"
#include <math.h>
#include <iostream>
#include <fstream>
#include <omp.h>

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
    input_file.read((char*)&(pointcloud2->height), sizeof(int32_t));
    input_file.read((char*)&(pointcloud2->width), sizeof(int32_t));
    input_file.read((char*)&(pointcloud2->point_step), sizeof(uint32_t));
    int pos = 0;
    pointcloud2->data = new float[pointcloud2->height * pointcloud2->width * pointcloud2->point_step];
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
  testcases = 0;
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

PointsImage
pointcloud2_to_image(const PointCloud2& pointcloud2,
                     const Mat44& cameraExtrinsicMat,
                     const Mat33& cameraMat, const Vec5& distCoeff,
                     const ImageSize& imageSize)
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
        int32_t max_y = -1;
        int32_t min_y = h;
        
        msg.image_height = imageSize.height;
        msg.image_width = imageSize.width;
       for (uint32_t y = 0; y < pointcloud2.height; ++y) {
        #pragma omp parallel for reduction(max : max_y) reduction(min : min_y) schedule(static)
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
                        
                        if (point.data[2] <= 2.5) {
                                continue;
                        }

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
                        if(0 <= px && px < w && 0 <= py && py < h)
                        {
                                int pid = py * w + px;
                                #pragma omp critical
                                {
                                  if(msg.distance[pid] == 0 ||
                                     msg.distance[pid] > (point.data[2] * 100.0))
                                  {
                                          msg.distance[pid] = float(point.data[2] * 100);
                                          msg.intensity[pid] = float(intensity);

                                          max_y = py > max_y ? py : max_y;
                                          min_y = py < min_y ? py : min_y;

                                  }
                                }
                                #pragma omp critical
                                {
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
        }

        msg.max_y = max_y;
        msg.min_y = min_y;
        return msg;
}


void points2image::run(int p) {
  pause_func();
  
  while (read_testcases < testcases)
    {
      int count = read_next_testcases(p);
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
